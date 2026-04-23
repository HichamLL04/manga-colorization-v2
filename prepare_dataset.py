"""
prepare_dataset.py — Crear pares sintéticos B/N → Color desde páginas de webtoon
==================================================================================

El problema: solo tienes páginas coloreadas, no el B/N original.
La solución: convertir el color a B/N de forma que imite cómo se ve el manga real.

Una simple conversión RGB→Gray NO funciona bien porque:
  - El manga real tiene líneas de tinta negra muy definidas sobre blanco
  - El manhwa coloreado tiene antialiasing, sombras de color, etc.
  - El modelo fue entrenado con manga de líneas limpias → confunde el dominio

Este script hace una conversión mejorada que:
  1. Extrae el canal de luminancia (más fiel al manga que RGB promedio)
  2. Aumenta contraste de líneas (simula la tinta negra del manga)
  3. Blanquea zonas claras (simula el papel blanco del manga)
  4. Aplica ligero ruido (simula el escaneo)
  5. Guarda los pares en color/ y bw/ listos para train.py

Uso:
    python prepare_dataset.py --input /ruta/webtoon_coloreado --output /ruta/dataset
    python prepare_dataset.py --input /ruta/webtoon --output /ruta/dataset --preview
"""

import os
import glob
import argparse
import random
import numpy as np
import cv2
from tqdm import tqdm


EXTENSIONS = ('*.jpg', '*.jpeg', '*.png', '*.webp')


def list_images(folder):
    files = []
    for ext in EXTENSIONS:
        files += glob.glob(os.path.join(folder, '**', ext), recursive=True)
        files += glob.glob(os.path.join(folder, ext))
    return sorted(set(files))


def color_to_manga_bw(color_img, args):
    """
    Convierte una imagen RGB coloreada a B/N imitando el estilo de manga escaneado.

    Parámetros ajustables:
      - line_boost:    qué tanto reforzar las líneas oscuras
      - white_thresh:  umbral para blanquear zonas claras (simula papel)
      - noise_amount:  cantidad de ruido de escaneo
    """
    # 1. Convertir a espacio LAB y extraer canal L (luminancia)
    lab   = cv2.cvtColor(color_img, cv2.COLOR_RGB2LAB)
    gray  = lab[:, :, 0]   # canal L, rango 0-255

    # 2. Aumentar contraste con CLAHE (resalta líneas de tinta)
    clahe = cv2.createCLAHE(clipLimit=args.clahe_clip, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)

    # 3. Reforzar líneas oscuras (zona de tinta negra del manga)
    #    Detectar bordes y mezclarlos con el gris
    if args.line_boost > 0:
        edges = cv2.Canny(gray, 80, 180)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        gray  = np.where(edges > 0,
                         np.clip(gray.astype(int) - args.line_boost * 60, 0, 255),
                         gray).astype(np.uint8)

    # 4. Blanquear zonas muy claras (papel blanco del manga)
    if args.white_thresh > 0:
        white_mask = gray > args.white_thresh
        gray[white_mask] = np.clip(
            gray[white_mask].astype(int) + 30, 0, 255
        ).astype(np.uint8)

    # 5. Ligero blur para suavizar (simula impresión)
    gray = cv2.GaussianBlur(gray, (3, 3), 0.5)

    # 6. Ruido de escaneo (opcional)
    if args.noise_amount > 0:
        noise = np.random.normal(0, args.noise_amount, gray.shape).astype(np.int16)
        gray  = np.clip(gray.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return gray


def augment_color(color_img, args):
    """
    Aumentación del color para simular variaciones de estilo.
    Ayuda a evitar que el modelo memorice colores exactos.
    """
    if not args.augment_color:
        return color_img

    img = color_img.copy().astype(np.float32)

    # Variación de brillo/contraste aleatoria
    alpha = random.uniform(0.85, 1.15)   # contraste
    beta  = random.uniform(-15, 15)       # brillo
    img   = np.clip(img * alpha + beta, 0, 255)

    # Variación de saturación leve
    hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] *= random.uniform(0.8, 1.2)
    hsv = np.clip(hsv, 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    return img.astype(np.uint8)


def resize_for_training(img, target_size, is_gray=False):
    """
    Redimensiona manteniendo proporción, asegurando que sea >= target_size
    en ambas dimensiones (para que el random crop funcione correctamente).
    """
    h, w = img.shape[:2]
    min_dim = min(h, w)
    if min_dim < target_size:
        scale = target_size / min_dim + 0.01
        new_w = int(w * scale)
        new_h = int(h * scale)
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4
        if is_gray:
            img = cv2.resize(img, (new_w, new_h), interpolation=interp)
        else:
            img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    return img


def process_dataset(args):
    os.makedirs(os.path.join(args.output, 'color'), exist_ok=True)
    os.makedirs(os.path.join(args.output, 'bw'),    exist_ok=True)

    images = list_images(args.input)
    if not images:
        raise FileNotFoundError(f"No se encontraron imágenes en: {args.input}")

    print(f"\nEncontradas {len(images)} imágenes en: {args.input}")
    print(f"Destino: {args.output}")
    print(f"Configuración:")
    print(f"  Tamaño mínimo:  {args.min_size}px")
    print(f"  CLAHE clip:     {args.clahe_clip}")
    print(f"  Line boost:     {args.line_boost}")
    print(f"  White thresh:   {args.white_thresh}")
    print(f"  Noise amount:   {args.noise_amount}")
    print(f"  Augment color:  {args.augment_color}")
    print(f"  Multiplicador:  x{args.multiply} (genera más pares por imagen)\n")

    n_saved  = 0
    n_failed = 0

    for img_path in tqdm(images, desc="Procesando"):
        color_bgr = cv2.imread(img_path)
        if color_bgr is None:
            print(f"  [SKIP] No se pudo leer: {img_path}")
            n_failed += 1
            continue

        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

        # Saltar imágenes demasiado pequeñas
        h, w = color_rgb.shape[:2]
        if h < args.min_size // 2 or w < args.min_size // 2:
            print(f"  [SKIP] Demasiado pequeña ({w}x{h}): {os.path.basename(img_path)}")
            n_failed += 1
            continue

        # Redimensionar si hace falta
        color_rgb = resize_for_training(color_rgb, args.min_size)

        # Convertir a B/N estilo manga
        bw = color_to_manga_bw(color_rgb, args)

        base_name = os.path.splitext(os.path.basename(img_path))[0]

        # Guardar par base
        cv2.imwrite(
            os.path.join(args.output, 'color', f'{base_name}.png'),
            cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
        )
        cv2.imwrite(
            os.path.join(args.output, 'bw', f'{base_name}.png'),
            bw
        )
        n_saved += 1

        # Multiplicar dataset con variaciones (flip + color aug)
        for m in range(1, args.multiply):
            aug_color = augment_color(color_rgb, args)
            aug_bw    = color_to_manga_bw(aug_color, args)

            # Flip horizontal aleatorio
            if random.random() > 0.5:
                aug_color = np.fliplr(aug_color)
                aug_bw    = np.fliplr(aug_bw)

            cv2.imwrite(
                os.path.join(args.output, 'color', f'{base_name}_aug{m}.png'),
                cv2.cvtColor(aug_color, cv2.COLOR_RGB2BGR)
            )
            cv2.imwrite(
                os.path.join(args.output, 'bw', f'{base_name}_aug{m}.png'),
                aug_bw
            )
            n_saved += 1

    print(f"\n✅ Dataset listo:")
    print(f"   Pares guardados: {n_saved}")
    print(f"   Fallidas/saltadas: {n_failed}")
    print(f"   Carpeta color/: {args.output}/color")
    print(f"   Carpeta bw/:    {args.output}/bw")

    # Preview de la conversión (guarda una comparación visual)
    if args.preview and images:
        make_preview(images[:3], args)


def make_preview(sample_paths, args):
    """Genera una imagen de comparación color vs B/N generado."""
    rows = []
    for p in sample_paths:
        bgr = cv2.imread(p)
        if bgr is None:
            continue
        rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb   = resize_for_training(rgb, args.min_size)
        bw    = color_to_manga_bw(rgb, args)
        bw_3  = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

        # Redimensionar ambos a la misma altura para apilar
        h     = min(rgb.shape[0], 600)
        scale = h / rgb.shape[0]
        new_w = int(rgb.shape[1] * scale)
        rgb_r = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (new_w, h))
        bw_r  = cv2.resize(bw_3, (new_w, h))

        separator = np.ones((h, 10, 3), dtype=np.uint8) * 200
        row = np.hstack([rgb_r, separator, bw_r])
        rows.append(row)

    if not rows:
        return

    # Igualar anchos
    max_w = max(r.shape[1] for r in rows)
    rows  = [np.hstack([r, np.ones((r.shape[0], max_w - r.shape[1], 3),
                                    dtype=np.uint8) * 255]) for r in rows]
    preview = np.vstack(rows)

    out_path = os.path.join(args.output, 'preview_conversion.png')
    cv2.imwrite(out_path, preview)
    print(f"\n🖼️  Preview guardado en: {out_path}")
    print("   (izquierda = color original, derecha = B/N generado)")


def parse_args():
    p = argparse.ArgumentParser(
        description="Preparar dataset sintético B/N→Color desde páginas de webtoon"
    )
    p.add_argument('--input',        required=True,
                   help="Carpeta con las imágenes coloreadas del webtoon")
    p.add_argument('--output',       required=True,
                   help="Carpeta destino (se crean subcarpetas color/ y bw/)")
    p.add_argument('--min_size',     type=int,   default=512,
                   help="Tamaño mínimo de lado para entrenamiento (default: 512)")
    p.add_argument('--multiply',     type=int,   default=3,
                   help="Generar N pares por imagen con augmentaciones (default: 3)")
    p.add_argument('--clahe_clip',   type=float, default=2.0,
                   help="Clip limit del CLAHE para contraste (default: 2.0)")
    p.add_argument('--line_boost',   type=float, default=1.0,
                   help="Refuerzo de líneas oscuras 0-2 (default: 1.0, 0=desactivado)")
    p.add_argument('--white_thresh', type=int,   default=210,
                   help="Umbral para blanquear zonas claras 0-255 (default: 210)")
    p.add_argument('--noise_amount', type=float, default=3.0,
                   help="Intensidad del ruido de escaneo (default: 3.0, 0=sin ruido)")
    p.add_argument('--augment_color',action='store_true',
                   help="Aplicar variaciones de color/brillo en las copias augmentadas")
    p.add_argument('--preview',      action='store_true',
                   help="Guardar imagen de comparación color vs B/N generado")
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    process_dataset(args)
