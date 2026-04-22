"""
train.py  -  Entrenamiento del Generator para Manga Colorization V2
=====================================================================

Arquitectura confirmada desde generator.pth (645 capas):
  - Input:  [bw(1ch) + hint(4ch)] = 5 canales  -> to0.0.weight [32, 5, 3, 3]
  - Encoder interno SEResNeXt: layer1(256ch) / layer2(512ch) / layer3(1024ch)
  - Rama auxiliar: to0->to1->to2->to3  (128ch) concatenada con x3(1024) en tunnel4
  - tunnel4 input: 1024 + 128 = 1152ch  (confirmado por peso [512, 1152, 3,3])
  - tunnel3 input: 256  + 512 = 768ch
  - tunnel2 input: 128  + 256 + 64 = 448ch
  - tunnel1 input: 64   + 32 = 96ch
  - exit    input: 64   + 32 = 96ch
  - Salidas: (main_output [B,3,H,W], guide_output [B,3,H/4,W/4])

Estructura de dataset esperada:
    <data_path>/
        color/   <- imágenes RGB (.jpg/.png)
        bw/      <- mismas en escala de grises (opcional; se genera on-the-fly)

Comandos de ejemplo:
    # Entrenamiento desde cero
    python train.py --path /ruta/dataset --gpu

    # Partir de los pesos pre-entrenados oficiales (fine-tuning)
    python train.py --path /ruta/dataset --gpu --pretrained networks/generator.zip

    # Fine-tuning congelando el encoder (más rápido, menos VRAM)
    python train.py --path /ruta/dataset --gpu --pretrained networks/generator.zip --freeze_encoder

    # Reanudar un entrenamiento interrumpido
    python train.py --path /ruta/dataset --gpu --resume checkpoint_epoch5.pth

    # Multi-GPU (usa todas las GPUs disponibles automáticamente)
    python train.py --path /ruta/dataset --gpu --batch_size 8

    # Guardar solo cada 5 epochs, conservar los 2 últimos checkpoints
    python train.py --path /ruta/dataset --gpu --save_every 5 --keep_last 2

    # Opciones avanzadas
    python train.py --path /ruta/dataset --gpu --epochs 30 --batch_size 4 \
                    --lr_gen 1e-4 --lr_disc 1e-4 --content_loss --crop_size 384
"""

import os
import glob
import argparse
import datetime
import random
import re

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as tvm
from torch.utils.data import Dataset, DataLoader

from networks.models import Generator, SpectralNorm


# ──────────────────────────────────────────────────────────────────────────────
# DISCRIMINADOR  (PatchGAN con Spectral Norm)
# ──────────────────────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    PatchGAN de 4 bloques con Spectral Normalization.
    Entrada: imagen RGB [B, 3, H, W]
    Salida: mapa de logits (no sigmoid — se usa BCEWithLogitsLoss)
    """
    def __init__(self, ndf=64):
        super().__init__()

        def sn_block(in_ch, out_ch, stride=2):
            return nn.Sequential(
                SpectralNorm(nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.net = nn.Sequential(
            sn_block(3,       ndf,     stride=2),
            sn_block(ndf,     ndf * 2, stride=2),
            sn_block(ndf * 2, ndf * 4, stride=2),
            sn_block(ndf * 4, ndf * 8, stride=1),
            SpectralNorm(nn.Conv2d(ndf * 8, 1, 4, stride=1, padding=1)),
        )

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# RED DE CONTENIDO VGG16 (perceptual loss, opcional)
# ──────────────────────────────────────────────────────────────────────────────

class ContentVGG(nn.Module):
    """Extrae features relu2_2 de VGG16 para perceptual loss."""
    def __init__(self):
        super().__init__()
        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(vgg.features.children())[:10])
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.features(x)


# ──────────────────────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────────────────────

EXTENSIONS = ('*.jpg', '*.jpeg', '*.png', '*.webp')

def _list_images(folder):
    files = []
    for ext in EXTENSIONS:
        files += glob.glob(os.path.join(folder, ext))
    return sorted(files)


class MangaDataset(Dataset):
    """
    Carga pares (bw, color) y genera hints de color aleatorios en tiempo de
    entrenamiento (simulando las pistas que da el usuario en inferencia).
    """
    def __init__(self, data_path, crop_size=512, augment=True,
                 hint_prob=0.7, max_hints=8, hint_radius=8):
        self.crop_size   = crop_size
        self.augment     = augment
        self.hint_prob   = hint_prob
        self.max_hints   = max_hints
        self.hint_radius = hint_radius

        color_dir = os.path.join(data_path, 'color')
        bw_dir    = os.path.join(data_path, 'bw')

        self.color_paths = _list_images(color_dir)
        if not self.color_paths:
            raise FileNotFoundError(f"No se encontraron imágenes en {color_dir}")

        self.bw_dir = bw_dir if os.path.isdir(bw_dir) else None
        print(f"[Dataset] {len(self.color_paths)} imágenes. "
              f"Carpeta bw: {'encontrada' if self.bw_dir else 'no existe -> conversión on-the-fly'}")

    def __len__(self):
        return len(self.color_paths)

    def _load_pair(self, idx):
        color_bgr = cv2.imread(self.color_paths[idx])
        if color_bgr is None:
            raise IOError(f"No se pudo leer: {self.color_paths[idx]}")
        color = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

        if self.bw_dir is not None:
            bw_path = os.path.join(self.bw_dir, os.path.basename(self.color_paths[idx]))
            bw = cv2.imread(bw_path, cv2.IMREAD_GRAYSCALE) if os.path.isfile(bw_path) \
                 else cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
        else:
            bw = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)

        return color, bw

    def _random_crop(self, color, bw):
        h, w = color.shape[:2]
        cs   = self.crop_size
        if h < cs:
            factor = cs / h + 1e-3
            color  = cv2.resize(color, (int(w * factor), cs), interpolation=cv2.INTER_AREA)
            bw     = cv2.resize(bw,    (int(w * factor), cs), interpolation=cv2.INTER_AREA)
            h, w   = color.shape[:2]
        if w < cs:
            factor = cs / w + 1e-3
            color  = cv2.resize(color, (cs, int(h * factor)), interpolation=cv2.INTER_AREA)
            bw     = cv2.resize(bw,    (cs, int(h * factor)), interpolation=cv2.INTER_AREA)
            h, w   = color.shape[:2]

        top  = random.randint(0, h - cs)
        left = random.randint(0, w - cs)
        return color[top:top+cs, left:left+cs], bw[top:top+cs, left:left+cs]

    def _make_hint(self, color_norm):
        h, w = color_norm.shape[:2]
        hint_rgb  = np.zeros((h, w, 3), dtype=np.float32)
        hint_mask = np.zeros((h, w, 1), dtype=np.float32)

        if random.random() < self.hint_prob:
            for _ in range(random.randint(1, self.max_hints)):
                cy, cx = random.randint(0, h-1), random.randint(0, w-1)
                r      = self.hint_radius
                y1, y2 = max(0, cy-r), min(h, cy+r)
                x1, x2 = max(0, cx-r), min(w, cx+r)
                avg = color_norm[y1:y2, x1:x2].mean(axis=(0, 1))
                hint_rgb[y1:y2, x1:x2]  = avg
                hint_mask[y1:y2, x1:x2] = 1.0

        hint = np.concatenate([hint_rgb * hint_mask, hint_mask], axis=2)
        return hint.transpose(2, 0, 1)

    def __getitem__(self, idx):
        color, bw = self._load_pair(idx)
        color, bw = self._random_crop(color, bw)

        if self.augment and random.random() < 0.5:
            color = np.fliplr(color).copy()
            bw    = np.fliplr(bw).copy()

        color_norm = (color.astype(np.float32) / 255.0 - 0.5) / 0.5
        bw_norm    = (bw.astype(np.float32)    / 255.0 - 0.5) / 0.5

        hint = self._make_hint(color_norm)

        color_t = torch.from_numpy(color_norm.transpose(2, 0, 1))
        bw_t    = torch.from_numpy(bw_norm[np.newaxis])
        hint_t  = torch.from_numpy(hint)

        gen_input = torch.cat([bw_t, hint_t], dim=0)

        return gen_input, color_t


# ──────────────────────────────────────────────────────────────────────────────
# FUNCIONES DE PÉRDIDA
# ──────────────────────────────────────────────────────────────────────────────

_bce = nn.BCEWithLogitsLoss()
_l1  = nn.L1Loss()
_mse = nn.MSELoss()

def compute_generator_loss(logits_fake, fake, guide, real,
                           content_net=None, w_sim=10.0, w_guide=0.9, w_content=1.0):
    sim   = _l1(fake, real) + w_guide * _l1(guide, real)
    adv   = _bce(logits_fake, torch.ones_like(logits_fake))
    total = w_sim * sim + adv

    if content_net is not None:
        feat_fake = content_net(fake)
        with torch.no_grad():
            feat_real = content_net(real)
        total = total + w_content * _mse(feat_fake, feat_real)

    return total

def compute_discriminator_loss(logits_real, logits_fake):
    real_loss = _bce(logits_real, torch.full_like(logits_real, 0.9))
    fake_loss = _bce(logits_fake, torch.zeros_like(logits_fake))
    return real_loss + fake_loss


# ──────────────────────────────────────────────────────────────────────────────
# GESTIÓN DE CHECKPOINTS
# ──────────────────────────────────────────────────────────────────────────────

def cleanup_old_checkpoints(output_dir, keep_last=1):
    """Borra checkpoints antiguos, conserva solo los últimos `keep_last`."""
    def sorted_by_epoch(pattern):
        files = [f for f in os.listdir(output_dir) if re.match(pattern, f)]
        return sorted(files, key=lambda f: int(re.search(r'\d+', f).group()))

    for pattern in [r'checkpoint_epoch\d+\.pth', r'generator_epoch\d+\.pth']:
        files = sorted_by_epoch(pattern)
        for f in files[:-keep_last]:
            path = os.path.join(output_dir, f)
            os.remove(path)
            print(f"  [Limpieza] Borrado: {f}")


# ──────────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    # ── Dispositivo y multi-GPU ──────────────────────────────────────────────
    use_cuda   = args.gpu and torch.cuda.is_available()
    device     = torch.device('cuda' if use_cuda else 'cpu')
    n_gpus     = torch.cuda.device_count() if use_cuda else 0

    print(f"[Train] Dispositivo: {device}")
    if args.gpu and not use_cuda:
        print("[Train] AVISO: --gpu activado pero CUDA no disponible. Usando CPU.")
    if n_gpus > 1:
        print(f"[Train] Multi-GPU activado: {n_gpus} GPUs detectadas")
    elif n_gpus == 1:
        print(f"[Train] 1 GPU detectada: {torch.cuda.get_device_name(0)}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset    = MangaDataset(args.path, crop_size=args.crop_size,
                               hint_prob=args.hint_prob, max_hints=args.max_hints)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.workers,
                            pin_memory=use_cuda, drop_last=True)

    # ── Modelos ───────────────────────────────────────────────────────────────
    generator     = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Multi-GPU: envolver con DataParallel
    if n_gpus > 1:
        generator     = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
        print(f"[Train] DataParallel activado en {n_gpus} GPUs")

    # Referencia al módulo real (necesario para acceder a .encoder, .state_dict, etc.)
    gen_module  = generator.module  if n_gpus > 1 else generator
    disc_module = discriminator.module if n_gpus > 1 else discriminator

    # ── Pesos pre-entrenados ──────────────────────────────────────────────────
    if args.pretrained:
        state = torch.load(args.pretrained, map_location=device)
        # Compatibilidad: si es un checkpoint completo, extraer solo el generator
        if isinstance(state, dict) and 'generator' in state:
            print("[Train] Detectado checkpoint completo — extrayendo pesos del generator")
            state = state['generator']
        gen_module.load_state_dict(state)
        print(f"[Train] Pesos del generator cargados: {args.pretrained}")

    # ── Content loss VGG (opcional) ───────────────────────────────────────────
    content_net = ContentVGG().to(device) if args.content_loss else None
    if content_net:
        print("[Train] Perceptual loss VGG16 activado")

    # ── Congelar encoder ──────────────────────────────────────────────────────
    if args.freeze_encoder:
        for p in gen_module.encoder.parameters():
            p.requires_grad = False
        print("[Train] Encoder SEResNeXt congelado")

    # ── Optimizadores ─────────────────────────────────────────────────────────
    gen_params = filter(lambda p: p.requires_grad, generator.parameters())
    optG = optim.Adam(gen_params,                 lr=args.lr_gen,  betas=(0.5, 0.9))
    optD = optim.Adam(discriminator.parameters(), lr=args.lr_disc, betas=(0.5, 0.9))

    # ── Reanudar checkpoint ───────────────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        gen_module.load_state_dict(ckpt['generator'])
        disc_module.load_state_dict(ckpt['discriminator'])
        optG.load_state_dict(ckpt['optG'])
        optD.load_state_dict(ckpt['optD'])
        start_epoch = ckpt['epoch'] + 1
        print(f"[Train] Reanudando desde epoch {start_epoch}")

    output_dir = os.path.dirname(args.resume) if args.resume else '.'

    # ── Bucle principal ───────────────────────────────────────────────────────
    disc_turn = True

    for epoch in range(start_epoch, args.epochs):

        if epoch == args.lr_decay_epoch > 0:
            for g in optG.param_groups: g['lr'] /= 10
            for g in optD.param_groups: g['lr'] /= 10
            print(f"[Train] LR reducido x10 en epoch {epoch}")

        generator.train()
        discriminator.train()

        sum_g, sum_d, n_g, n_d = 0.0, 0.0, 0, 0

        for step, (gen_input, real_color) in enumerate(dataloader):
            gen_input  = gen_input.to(device)
            real_color = real_color.to(device)

            if disc_turn:
                for p in discriminator.parameters(): p.requires_grad = True
                for p in generator.parameters():     p.requires_grad = False
                discriminator.zero_grad()

                with torch.no_grad():
                    fake_color, _ = generator(gen_input)

                d_loss = compute_discriminator_loss(
                    discriminator(real_color),
                    discriminator(fake_color.detach())
                )
                d_loss.backward()
                optD.step()
                sum_d += d_loss.item(); n_d += 1

            else:
                for p in discriminator.parameters(): p.requires_grad = False
                for p in generator.parameters():     p.requires_grad = True
                generator.zero_grad()

                fake_color, guide_color = generator(gen_input)
                g_loss = compute_generator_loss(
                    discriminator(fake_color),
                    fake_color, guide_color, real_color,
                    content_net=content_net
                )
                g_loss.backward()
                optG.step()
                sum_g += g_loss.item(); n_g += 1

            disc_turn = not disc_turn

            if step % 50 == 0:
                ts = datetime.datetime.now().strftime('%H:%M:%S')
                print(f"  [{ts}] Epoch {epoch:3d}  Step {step:5d}  "
                      f"D: {sum_d/(n_d+1e-8):.4f}  G: {sum_g/(n_g+1e-8):.4f}")

        avg_d = sum_d / (n_d + 1e-8)
        avg_g = sum_g / (n_g + 1e-8)
        print(f"\n[Epoch {epoch}] D: {avg_d:.4f}  G: {avg_g:.4f}")

        # ── Guardar cada save_every epochs (y siempre el último) ─────────────
        is_last  = (epoch == args.epochs - 1)
        do_save  = (epoch % args.save_every == 0) or is_last

        if do_save:
            ckpt_path = os.path.join(output_dir, f"checkpoint_epoch{epoch}.pth")
            gen_path  = os.path.join(output_dir, f"generator_epoch{epoch}.pth")

            torch.save({
                'epoch':         epoch,
                'generator':     gen_module.state_dict(),
                'discriminator': disc_module.state_dict(),
                'optG':          optG.state_dict(),
                'optD':          optD.state_dict(),
            }, ckpt_path)

            torch.save(gen_module.state_dict(), gen_path)

            print(f"  Guardado: checkpoint_epoch{epoch}.pth  |  generator_epoch{epoch}.pth")

            # Limpiar checkpoints antiguos
            cleanup_old_checkpoints(output_dir, keep_last=args.keep_last)
        else:
            print(f"  [Checkpoint omitido — próximo guardado en epoch "
                  f"{epoch + (args.save_every - epoch % args.save_every)}]")

        print()

    print("[Train] Entrenamiento completado.")


# ──────────────────────────────────────────────────────────────────────────────
# ARGUMENTOS
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Entrenamiento Manga Colorization V2")
    p.add_argument('--path',           required=True,           help="Ruta al dataset (con subcarpeta color/)")
    p.add_argument('--gpu',            action='store_true',     help="Usar CUDA (multi-GPU automático si hay varias)")
    p.add_argument('--epochs',         type=int,   default=20,  help="Epochs (default: 20)")
    p.add_argument('--batch_size',     type=int,   default=2,   help="Batch size TOTAL entre todas las GPUs (default: 2)")
    p.add_argument('--crop_size',      type=int,   default=512, help="Crop cuadrado, divisible por 32 (default: 512)")
    p.add_argument('--lr_gen',         type=float, default=2e-4,help="LR generador (default: 2e-4)")
    p.add_argument('--lr_disc',        type=float, default=2e-4,help="LR discriminador (default: 2e-4)")
    p.add_argument('--lr_decay_epoch', type=int,   default=-1,  help="Epoch donde se divide LR por 10 (-1=nunca)")
    p.add_argument('--workers',        type=int,   default=4,   help="DataLoader workers (default: 4)")
    p.add_argument('--hint_prob',      type=float, default=0.7, help="Probabilidad de hints de color (default: 0.7)")
    p.add_argument('--max_hints',      type=int,   default=8,   help="Máximo de puntos de hint (default: 8)")
    p.add_argument('--pretrained',     type=str,   default='',  help="generator.pth/.zip para partir de pesos existentes")
    p.add_argument('--resume',         type=str,   default='',  help="checkpoint_epochN.pth para reanudar")
    p.add_argument('--freeze_encoder', action='store_true',     help="Congelar encoder SEResNeXt (fine-tuning)")
    p.add_argument('--content_loss',   action='store_true',     help="Activar perceptual loss VGG16")
    # ── Nuevos argumentos ────────────────────────────────────────────────────
    p.add_argument('--save_every',     type=int,   default=1,   help="Guardar checkpoint cada N epochs (default: 1)")
    p.add_argument('--keep_last',      type=int,   default=1,   help="Número de checkpoints a conservar (default: 1)")
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.crop_size % 32 != 0:
        raise ValueError(f"--crop_size debe ser divisible por 32, recibido: {args.crop_size}")
    train(args)
