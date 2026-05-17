import torch
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import numpy as np
import cv2

from networks.models import Colorizer
from denoising.denoiser import FFDNetDenoiser
from utils.utils import resize_pad


def generate_distance_field_map(image):
    try:
        import snowy
    except ImportError as error:
        raise RuntimeError('snowy is required to generate Distance Field Maps') from error

    if len(image.shape) == 3:
        image = image[:, :, 0]

    if image.max() <= 1.2:
        image = image * 255

    image = np.clip(image, 0, 255).astype(np.uint8)
    _, binary = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dfm = snowy.generate_sdf(binary.astype(bool)).astype(np.float32)
    dfm_min = dfm.min()
    dfm_max = dfm.max()

    if dfm_max > dfm_min:
        dfm = (dfm - dfm_min) / (dfm_max - dfm_min)
    else:
        dfm = np.zeros_like(dfm, dtype=np.float32)

    return dfm


def detect_generator_input_channels(state_dict):
    for key, weights in state_dict.items():
        if key.endswith('to0.0.weight'):
            return weights.shape[1]

    return 5


class MangaColorizator:
    def __init__(self, device, generator_path = 'networks/generator.zip', extractor_path = 'networks/extractor.pth'):
        generator_state = torch.load(generator_path, map_location = device)
        self.generator_input_channels = detect_generator_input_channels(generator_state)
        if self.generator_input_channels not in (5, 6):
            raise RuntimeError('unsupported generator input channels: {}'.format(self.generator_input_channels))
        self.colorizer = Colorizer(input_channels=self.generator_input_channels).to(device)
        self.colorizer.generator.load_state_dict(generator_state)
        self.colorizer = self.colorizer.eval()
        self.use_dfm = self.generator_input_channels == 6
        
        self.denoiser = FFDNetDenoiser(device)
        
        self.current_image = None
        self.current_hint = None
        self.current_pad = None
        
        self.device = device
        
    def set_image(self, image, size = 576, apply_denoise = True, denoise_sigma = 25, transform = ToTensor()):
        if (size % 32 != 0):
            raise RuntimeError("size is not divisible by 32")
        
        if apply_denoise:
            image = self.denoiser.get_denoised_image(image, sigma = denoise_sigma)
        
        image, self.current_pad = resize_pad(image, size)
        image_tensor = transform(image).float()
        if self.use_dfm:
            dfm = generate_distance_field_map(image)
            dfm_tensor = torch.from_numpy(dfm).unsqueeze(0).float()
            image_tensor = torch.cat([image_tensor, dfm_tensor], 0)

        self.current_image = image_tensor.unsqueeze(0).to(self.device)
        self.current_hint = torch.zeros(1, 4, self.current_image.shape[2], self.current_image.shape[3]).float().to(self.device)
    
    def update_hint(self, hint, mask):
        '''
        Args:
           hint: numpy.ndarray with shape (self.current_image.shape[2], self.current_image.shape[3], 3)
           mask: numpy.ndarray with shape (self.current_image.shape[2], self.current_image.shape[3])
        '''
        
        if issubclass(hint.dtype.type, np.integer):
            hint = hint.astype('float32') / 255
            
        hint = (hint - 0.5) / 0.5
        hint = torch.FloatTensor(hint).permute(2, 0, 1)
        mask = torch.FloatTensor(np.expand_dims(mask, 0))

        self.current_hint = torch.cat([hint * mask, mask], 0).unsqueeze(0).to(self.device)

    def _predict(self):
        with torch.no_grad():
            fake_color, _ = self.colorizer(torch.cat([self.current_image, self.current_hint], 1))
            return fake_color.detach()

    def _tensor_to_image(self, tensor, pad, crop=True):
        result = tensor.detach().cpu().permute(1, 2, 0) * 0.5 + 0.5

        if crop:
            if pad[0] != 0:
                result = result[:-pad[0]]
            if pad[1] != 0:
                result = result[:, :-pad[1]]
            
        return result.numpy()

    def colorize(self):
        fake_color = self._predict()
        return self._tensor_to_image(fake_color[0], self.current_pad)

    def colorize_with_autohint(self):
        self.current_hint.zero_()
        initial_color = self._predict()
        dense_hint = self._tensor_to_image(initial_color[0], self.current_pad, crop=False)
        dense_mask = np.ones(dense_hint.shape[:2], dtype=np.float32)
        self.update_hint(dense_hint, dense_mask)
        return self.colorize()

    def colorize_batch(self, images, hints, pads):
        with torch.no_grad():
            fake_color, _ = self.colorizer(torch.cat([images, hints], 1))
            fake_color = fake_color.detach()

        results = []
        for index, image in enumerate(fake_color):
            results.append(self._tensor_to_image(image, pads[index]))

        return results

    def colorize_batch_with_autohint(self, images, hints, pads):
        hints.zero_()
        with torch.no_grad():
            initial_color, _ = self.colorizer(torch.cat([images, hints], 1))
            dense_hints = []

            for image in initial_color.detach():
                dense_hint = image.detach().cpu() * 0.5 + 0.5
                dense_mask = torch.ones(1, dense_hint.shape[1], dense_hint.shape[2])
                dense_hints.append(torch.cat([(dense_hint - 0.5) / 0.5 * dense_mask, dense_mask], 0))

            dense_hints = torch.stack(dense_hints).to(self.device)
            final_color, _ = self.colorizer(torch.cat([images, dense_hints], 1))
            final_color = final_color.detach()

        return [self._tensor_to_image(image, pads[index]) for index, image in enumerate(final_color)]

    def pad_batch(self, images, hints, pads):
        max_height = max(image.shape[1] for image in images)
        max_width = max(image.shape[2] for image in images)
        padded_images = []
        padded_hints = []
        padded_pads = []

        for image, hint, pad in zip(images, hints, pads):
            extra_h = max_height - image.shape[1]
            extra_w = max_width - image.shape[2]
            padded_images.append(F.pad(image, (0, extra_w, 0, extra_h), mode='replicate'))
            padded_hints.append(F.pad(hint, (0, extra_w, 0, extra_h)))
            padded_pads.append((pad[0] + extra_h, pad[1] + extra_w))

        return torch.stack(padded_images).to(self.device), torch.stack(padded_hints).to(self.device), padded_pads
