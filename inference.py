import argparse
import logging
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from tqdm import tqdm

from colorizator import MangaColorizator, generate_distance_field_map
from utils.utils import resize_pad


IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')
PALETTE = {
    'red': (220, 40, 40),
    'blue': (45, 100, 220),
    'green': (50, 170, 85),
    'yellow': (235, 210, 60),
    'orange': (235, 130, 40),
    'purple': (145, 85, 200),
    'pink': (235, 110, 170),
    'brown': (140, 95, 55),
}


class MangaImageDataset(Dataset):
    def __init__(self, image_paths, size, apply_denoise, denoise_sigma, denoiser, include_dfm):
        self.image_paths = image_paths
        self.size = size
        self.apply_denoise = apply_denoise
        self.denoise_sigma = denoise_sigma
        self.denoiser = denoiser
        self.include_dfm = include_dfm
        self.transform = ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        try:
            image = plt.imread(image_path)
            original = ensure_rgb(image)

            if self.apply_denoise:
                image = self.denoiser.get_denoised_image(image, sigma=self.denoise_sigma)

            image, pad = resize_pad(image, self.size)
            tensor = self.transform(image).float()
            if self.include_dfm:
                dfm = generate_distance_field_map(image)
                tensor = torch.cat([tensor, torch.from_numpy(dfm).unsqueeze(0).float()], 0)
            hint = torch.zeros(4, tensor.shape[1], tensor.shape[2]).float()

            return {
                'path': image_path,
                'image': tensor,
                'hint': hint,
                'pad': pad,
                'original': original,
                'error': None,
            }
        except Exception as error:
            return {'path': image_path, 'error': str(error)}


def collate_images(batch):
    successes = [item for item in batch if item.get('error') is None]
    failures = [item for item in batch if item.get('error') is not None]
    return successes, failures


def process_image(image, colorizator, args):
    colorizator.set_image(image, args.size, args.denoiser, args.denoiser_sigma)
    if args.interactive_hints:
        hint, mask = collect_interactive_hint(colorizator.current_image, args.hint_threshold)
        colorizator.update_hint(hint, mask)

    if args.autohint:
        return colorizator.colorize_with_autohint()

    return colorizator.colorize()


def colorize_single_image(image_path, save_path, colorizator, args):
    image = plt.imread(image_path)
    colorization = process_image(image, colorizator, args)
    save_result(image, colorization, save_path, args.compare)
    return True


def colorize_images(target_path, colorizator, args):
    image_paths = get_image_paths(args.path)
    failures = []

    if args.interactive_hints:
        for image_path in tqdm(image_paths, desc='Colorizando', unit='img'):
            try:
                save_path = build_save_path(target_path, image_path)
                colorize_single_image(image_path, save_path, colorizator, args)
            except Exception as error:
                log_failure(failures, image_path, error)
    else:
        dataset = MangaImageDataset(
            image_paths,
            args.size,
            args.denoiser,
            args.denoiser_sigma,
            colorizator.denoiser,
            colorizator.use_dfm,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=collate_images,
        )

        for items, failed_items in tqdm(loader, desc='Colorizando', unit='batch'):
            for item in failed_items:
                log_failure(failures, item['path'], item['error'])

            if not items:
                continue

            try:
                images, hints, pads = colorizator.pad_batch(
                    [item['image'] for item in items],
                    [item['hint'] for item in items],
                    [item['pad'] for item in items],
                )
                if args.autohint:
                    colorizations = colorizator.colorize_batch_with_autohint(images, hints, pads)
                else:
                    colorizations = colorizator.colorize_batch(images, hints, pads)
                for item, colorization in zip(items, colorizations):
                    save_path = build_save_path(target_path, item['path'])
                    save_result(item['original'], colorization, save_path, args.compare)
            except RuntimeError as error:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                for item in items:
                    log_failure(failures, item['path'], error)

    write_failure_log(target_path, failures)


def collect_interactive_hint(current_image, threshold):
    image = current_image[0, 0].detach().cpu().numpy()
    image_u8 = np.clip(image * 255, 0, 255).astype(np.uint8)
    hint = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
    mask = np.zeros(image.shape[:2], dtype=np.float32)
    overlay = np.repeat(image[:, :, None], 3, axis=2)
    selected = {'name': next(iter(PALETTE)), 'rgb': next(iter(PALETTE.values()))}

    fig, (image_axis, palette_axis) = plt.subplots(1, 2, width_ratios=[4, 1])
    image_artist = image_axis.imshow(overlay)
    image_axis.set_title('Click en la imagen para pintar; cierra la ventana al terminar')
    image_axis.axis('off')

    palette = np.array(list(PALETTE.values()), dtype=np.float32).reshape(len(PALETTE), 1, 3) / 255
    palette_axis.imshow(palette)
    palette_axis.set_yticks(range(len(PALETTE)))
    palette_axis.set_yticklabels(PALETTE.keys())
    palette_axis.set_xticks([])
    palette_axis.set_title('Color')

    def on_click(event):
        if event.inaxes == palette_axis and event.ydata is not None:
            index = int(np.clip(round(event.ydata), 0, len(PALETTE) - 1))
            name = list(PALETTE.keys())[index]
            selected['name'] = name
            selected['rgb'] = PALETTE[name]
            fig.suptitle('Color seleccionado: {}'.format(name))
            fig.canvas.draw_idle()
            return

        if event.inaxes != image_axis or event.xdata is None or event.ydata is None:
            return

        x = int(np.clip(round(event.xdata), 0, image.shape[1] - 1))
        y = int(np.clip(round(event.ydata), 0, image.shape[0] - 1))
        region = flood_fill_mask(image_u8, x, y, threshold)
        color = np.array(selected['rgb'], dtype=np.float32) / 255
        hint[region] = color
        mask[region] = 1.0
        overlay[region] = 0.55 * overlay[region] + 0.45 * color
        image_artist.set_data(overlay)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()
    return hint, mask


def flood_fill_mask(image, x, y, threshold):
    flood_mask = np.zeros((image.shape[0] + 2, image.shape[1] + 2), dtype=np.uint8)
    filled = image.copy()
    cv2.floodFill(filled, flood_mask, (x, y), 255, threshold, threshold)
    return flood_mask[1:-1, 1:-1].astype(bool)


def save_result(original, colorization, save_path, compare):
    if compare:
        plt.imsave(save_path, make_comparison(original, colorization))
    else:
        plt.imsave(save_path, colorization)


def make_comparison(original, colorization):
    original = ensure_rgb(original)
    if original.max() > 1.2:
        original = original.astype(np.float32) / 255
    original = cv2.resize(
        original,
        (colorization.shape[1], colorization.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    return np.concatenate([original, colorization], axis=1)


def ensure_rgb(image):
    if len(image.shape) == 2:
        image = np.expand_dims(image, 2)
    if image.shape[2] == 1:
        image = np.repeat(image, 3, 2)
    return image[:, :, :3]


def build_save_path(target_path, image_path):
    image_name = os.path.basename(image_path)
    name, ext = os.path.splitext(image_name)
    if ext.lower() != '.png':
        image_name = name + '.png'
    return os.path.join(target_path, image_name)


def get_image_paths(path):
    image_paths = []
    for image_name in sorted(os.listdir(path)):
        file_path = os.path.join(path, image_name)
        if os.path.isfile(file_path) and os.path.splitext(image_name)[1].lower() in IMAGE_EXTENSIONS:
            image_paths.append(file_path)
    return image_paths


def log_failure(failures, image_path, error):
    message = '{}: {}'.format(image_path, error)
    logging.error(message)
    failures.append(message)


def write_failure_log(target_path, failures):
    if not failures:
        return

    log_path = os.path.join(target_path, 'failures.log')
    with open(log_path, 'w') as log_file:
        log_file.write('\n'.join(failures))
        log_file.write('\n')


def validate_args(args):
    if args.size % 32 != 0:
        raise ValueError('--size debe ser multiplo de 32. Rango recomendado: 384-768 segun la memoria disponible.')
    if args.size < 32:
        raise ValueError('--size debe ser positivo y multiplo de 32. Rango recomendado: 384-768.')
    if args.batch_size < 1:
        raise ValueError('--batch_size debe ser al menos 1.')
    if args.workers < 0:
        raise ValueError('--workers no puede ser negativo.')
    if args.interactive_hints and args.batch_size != 1:
        print('Aviso: --interactive_hints procesa una imagen por vez; --batch_size se ignora.', file=sys.stderr)
    if args.interactive_hints and args.autohint:
        raise ValueError('--autohint no se puede combinar con --interactive_hints porque reemplaza los hints manuales.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', required=True)
    parser.add_argument('-gen', '--generator', default='networks/generator.zip')
    parser.add_argument('-ext', '--extractor', default='networks/extractor.pth')
    parser.add_argument('-g', '--gpu', dest='gpu', action='store_true')
    parser.add_argument('-nd', '--no_denoise', dest='denoiser', action='store_false')
    parser.add_argument('-ds', '--denoiser_sigma', type=int, default=25)
    parser.add_argument(
        '-s',
        '--size',
        type=int,
        default=576,
        help='Tamano base de inferencia, multiplo de 32. Rango recomendado: 384-768 segun memoria disponible.',
    )
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='Numero de imagenes por batch.')
    parser.add_argument('--workers', type=int, default=0, help='Workers de DataLoader para carga/preprocesado.')
    parser.add_argument('--compare', action='store_true', help='Guarda original BN y colorizacion lado a lado.')
    parser.add_argument('--interactive_hints', action='store_true', help='Activa clicks por region y paleta basica.')
    parser.add_argument('--hint_threshold', type=int, default=10, help='Tolerancia de flood fill para hints interactivos.')
    parser.add_argument('--autohint', action='store_true', help='Ejecuta una segunda pasada con hint denso generado desde la primera colorizacion.')
    parser.set_defaults(gpu=False)
    parser.set_defaults(denoiser=True)
    args = parser.parse_args()
    validate_args(args)
    return args


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')
    try:
        args = parse_args()
    except ValueError as error:
        print('Error: {}'.format(error), file=sys.stderr)
        sys.exit(2)

    device = 'cuda' if args.gpu else 'cpu'
    colorizer = MangaColorizator(device, args.generator, args.extractor)

    if os.path.isdir(args.path):
        colorization_path = os.path.join(args.path, 'colorization')
        if not os.path.exists(colorization_path):
            os.makedirs(colorization_path)

        colorize_images(colorization_path, colorizer, args)
    elif os.path.isfile(args.path):
        split = os.path.splitext(args.path)

        if split[1].lower() in IMAGE_EXTENSIONS:
            new_image_path = split[0] + '_colorized' + '.png'
            colorize_single_image(args.path, new_image_path, colorizer, args)
        else:
            print('Wrong format')
    else:
        print('Wrong path')
