import argparse
import hashlib
import random
from pathlib import Path

import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def page_output_paths(dataset_root, pair):
    base_name = pair.pair_id + '.png'
    return (
        Path(dataset_root) / 'pages' / 'bw' / base_name,
        Path(dataset_root) / 'pages' / 'color' / base_name,
    )


def prepare_pages(args):
    from dataset_pipeline.alignment import align_pair, fallback_resize_pair
    from dataset_pipeline.common import load_json, read_image, write_image, write_json
    from dataset_pipeline.matching import match_pages
    from dataset_pipeline.text_removal import load_detections_for_path, remove_text_boxes

    dataset_root = Path(args.output)
    text_detections = load_json(args.text_detections, default={}) if args.text_detections else {}
    pairs = match_pages(args.bw_root, args.color_root)
    metadata = {
        'pages': [],
        'panels': [],
        'splits': {},
        'config': vars(args),
    }

    for pair in tqdm(pairs, desc='Preparing pages', unit='page'):
        try:
            bw_image = read_image(pair.bw_path)
            color_image = read_image(pair.color_path)
            bw_boxes = load_detections_for_path(text_detections, pair.bw_path)
            color_boxes = load_detections_for_path(text_detections, pair.color_path)
            bw_clean, bw_removed = remove_text_boxes(bw_image, bw_boxes, args.text_variance)
            color_clean, color_removed = remove_text_boxes(color_image, color_boxes, args.text_variance)
            aligned_color, align_meta = align_pair(
                bw_clean,
                color_clean,
                min_matches=args.min_matches,
                ratio_test=args.ratio_test,
                ransac_threshold=args.ransac_threshold,
            )

            if aligned_color is None and args.allow_resize_fallback:
                aligned_color, align_meta = fallback_resize_pair(bw_clean, color_clean)

            if aligned_color is None or align_meta['inlier_ratio'] < args.min_inlier_ratio:
                continue

            bw_out, color_out = page_output_paths(dataset_root, pair)
            write_image(bw_out, bw_clean)
            write_image(color_out, aligned_color)
            metadata['pages'].append({
                'pair_id': pair.pair_id,
                'title': pair.title,
                'chapter': pair.chapter,
                'page': pair.page,
                'bw_path': str(bw_out.relative_to(dataset_root)),
                'color_path': str(color_out.relative_to(dataset_root)),
                'source_bw_path': str(pair.bw_path),
                'source_color_path': str(pair.color_path),
                'alignment': align_meta,
                'removed_text_boxes': {
                    'bw': bw_removed,
                    'color': color_removed,
                },
            })
        except Exception as error:
            metadata.setdefault('failures', []).append({'pair_id': pair.pair_id, 'error': str(error)})

    write_json(dataset_root / 'metadata.json', metadata)


def extract_panels(args):
    from dataset_pipeline.common import load_json, read_image, write_image, write_json
    from dataset_pipeline.panels import (
        SamPanelSegmenter,
        apply_panel_mask,
        boxes_from_detections,
        extract_panel_pair,
        find_panel_boxes,
        panel_pair_score,
    )

    dataset_root = Path(args.dataset)
    metadata = load_json(dataset_root / 'metadata.json', default={'pages': [], 'panels': [], 'splits': {}})
    panel_detections = load_json(args.panel_detections, default={}) if args.panel_detections else {}
    panels = []
    sam = None
    if getattr(args, 'sam_checkpoint', None):
        sam = SamPanelSegmenter(args.sam_checkpoint, args.sam_model_type, args.device)

    for page in tqdm(metadata.get('pages', []), desc='Extracting panels', unit='page'):
        bw_image = read_image(dataset_root / page['bw_path'])
        color_image = read_image(dataset_root / page['color_path'])
        detection_value = (
            panel_detections.get(page['pair_id'])
            or panel_detections.get(page['bw_path'])
            or panel_detections.get(Path(page['bw_path']).name)
        )
        boxes = boxes_from_detections(detection_value)
        if not boxes:
            boxes = find_panel_boxes(bw_image, args.min_panel_area)
        masks = sam.segment_boxes(bw_image, boxes) if sam else [None] * len(boxes)

        for panel_index, (box, mask) in enumerate(zip(boxes, masks)):
            bw_panel, color_panel, clipped = extract_panel_pair(bw_image, color_image, box)
            if mask is not None:
                mask_crop, _, _ = extract_panel_pair(mask, mask, clipped)
                bw_panel = apply_panel_mask(bw_panel, mask_crop)
                color_panel = apply_panel_mask(color_panel, mask_crop)
            score = panel_pair_score(bw_panel, color_panel)
            if score < args.min_ssim:
                continue

            panel_id = '{}_{:03d}'.format(page['pair_id'], panel_index)
            panel_name = panel_id + '.png'
            bw_out = dataset_root / 'panels' / 'bw' / panel_name
            color_out = dataset_root / 'panels' / 'color' / panel_name
            write_image(bw_out, bw_panel)
            write_image(color_out, color_panel)
            panels.append({
                'panel_id': panel_id,
                'page_id': page['pair_id'],
                'title': page['title'],
                'chapter': page['chapter'],
                'page': page['page'],
                'panel_index': panel_index,
                'bbox': clipped,
                'ssim': score,
                'bw_path': str(bw_out.relative_to(dataset_root)),
                'color_path': str(color_out.relative_to(dataset_root)),
            })

    metadata['panels'] = panels
    write_json(dataset_root / 'metadata.json', metadata)


def caption_panels(args):
    from dataset_pipeline.captions import merge_captions
    from dataset_pipeline.common import load_json, write_json

    dataset_root = Path(args.dataset)
    metadata = load_json(dataset_root / 'metadata.json', default={'panels': []})
    caption_sources = args.caption_sources or []

    for panel in tqdm(metadata.get('panels', []), desc='Writing captions', unit='panel'):
        caption = merge_captions(panel['panel_id'], caption_sources)
        caption_path = dataset_root / 'panels' / 'captions' / (panel['panel_id'] + '.json')
        write_json(caption_path, caption)
        panel['caption_path'] = str(caption_path.relative_to(dataset_root))

    write_json(dataset_root / 'metadata.json', metadata)


def build_splits(args):
    from dataset_pipeline.common import load_json, write_json

    dataset_root = Path(args.dataset)
    metadata = load_json(dataset_root / 'metadata.json', default={'panels': []})
    panels = metadata.get('panels', [])
    titles = sorted({panel['title'] for panel in panels})
    random.seed(args.seed)
    random.shuffle(titles)
    out_count = max(1, int(round(len(titles) * args.test_out_ratio))) if titles else 0
    test_out_titles = set(titles[:out_count])
    train = []
    test_in = []
    test_out = []

    for panel in panels:
        if panel['title'] in test_out_titles:
            test_out.append(panel['panel_id'])
            continue
        key = '{}:{}:{}'.format(args.seed, panel['title'], panel['chapter'])
        bucket = int(hashlib.sha1(key.encode('utf-8')).hexdigest(), 16) % 10000 / 10000
        if bucket < args.test_in_ratio:
            test_in.append(panel['panel_id'])
        else:
            train.append(panel['panel_id'])

    metadata['splits'] = {
        'train': sorted(train),
        'test_in': sorted(test_in),
        'test_out': sorted(test_out),
        'test_out_titles': sorted(test_out_titles),
    }
    write_json(dataset_root / 'metadata.json', metadata)


def run_all(args):
    prepare_pages(args)
    panel_args = argparse.Namespace(
        dataset=args.output,
        panel_detections=args.panel_detections,
        min_panel_area=args.min_panel_area,
        min_ssim=args.min_ssim,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        device=args.device,
    )
    extract_panels(panel_args)
    caption_args = argparse.Namespace(dataset=args.output, caption_sources=args.caption_sources)
    caption_panels(caption_args)
    split_args = argparse.Namespace(
        dataset=args.output,
        seed=args.seed,
        test_in_ratio=args.test_in_ratio,
        test_out_ratio=args.test_out_ratio,
    )
    build_splits(split_args)


def add_prepare_args(parser):
    parser.add_argument('--bw-root', required=True)
    parser.add_argument('--color-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--text-detections')
    parser.add_argument('--text-variance', type=float, default=300.0)
    parser.add_argument('--min-matches', type=int, default=24)
    parser.add_argument('--ratio-test', type=float, default=0.75)
    parser.add_argument('--ransac-threshold', type=float, default=5.0)
    parser.add_argument('--min-inlier-ratio', type=float, default=0.15)
    parser.add_argument('--allow-resize-fallback', action='store_true')


def parse_args():
    parser = argparse.ArgumentParser(description='Build an aligned paired manga colorization dataset.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    prepare = subparsers.add_parser('prepare-pages')
    add_prepare_args(prepare)
    prepare.set_defaults(func=prepare_pages)

    panels = subparsers.add_parser('extract-panels')
    panels.add_argument('--dataset', required=True)
    panels.add_argument('--panel-detections')
    panels.add_argument('--min-panel-area', type=float, default=0.01)
    panels.add_argument('--min-ssim', type=float, default=0.55)
    panels.add_argument('--sam-checkpoint')
    panels.add_argument('--sam-model-type', default='vit_h')
    panels.add_argument('--device', default='cuda')
    panels.set_defaults(func=extract_panels)

    captions = subparsers.add_parser('caption-panels')
    captions.add_argument('--dataset', required=True)
    captions.add_argument('--caption-sources', nargs='*')
    captions.set_defaults(func=caption_panels)

    splits = subparsers.add_parser('build-splits')
    splits.add_argument('--dataset', required=True)
    splits.add_argument('--seed', type=int, default=1337)
    splits.add_argument('--test-in-ratio', type=float, default=0.1)
    splits.add_argument('--test-out-ratio', type=float, default=0.15)
    splits.set_defaults(func=build_splits)

    all_parser = subparsers.add_parser('run-all')
    add_prepare_args(all_parser)
    all_parser.add_argument('--panel-detections')
    all_parser.add_argument('--min-panel-area', type=float, default=0.01)
    all_parser.add_argument('--min-ssim', type=float, default=0.55)
    all_parser.add_argument('--sam-checkpoint')
    all_parser.add_argument('--sam-model-type', default='vit_h')
    all_parser.add_argument('--device', default='cuda')
    all_parser.add_argument('--caption-sources', nargs='*')
    all_parser.add_argument('--seed', type=int, default=1337)
    all_parser.add_argument('--test-in-ratio', type=float, default=0.1)
    all_parser.add_argument('--test-out-ratio', type=float, default=0.15)
    all_parser.set_defaults(func=run_all)

    magi = subparsers.add_parser('detect-magi')
    magi.add_argument('--image-root', required=True)
    magi.add_argument('--output', required=True)
    magi.add_argument('--model', default='ragavsachdeva/magi')
    magi.add_argument('--batch-size', type=int, default=1)
    magi.add_argument('--device', default='cuda')
    magi.set_defaults(func=detect_magi)

    return parser.parse_args()


def read_magi_image(path):
    from PIL import Image

    with Path(path).open('rb') as file:
        image = Image.open(file).convert('L').convert('RGB')
        return np.array(image)


def detect_magi(args):
    import torch
    from transformers import AutoModel
    from dataset_pipeline.common import iter_images, write_json

    model = AutoModel.from_pretrained(args.model, trust_remote_code=True).to(args.device).eval()
    paths = list(iter_images(args.image_root))
    detections = {}

    for offset in tqdm(range(0, len(paths), args.batch_size), desc='Running MAGI', unit='batch'):
        batch_paths = paths[offset:offset + args.batch_size]
        images = [read_magi_image(path) for path in batch_paths]
        with torch.no_grad():
            results = model.predict_detections_and_associations(images)

        for path, result in zip(batch_paths, results):
            relative = str(Path(path).relative_to(args.image_root))
            detections[relative] = {
                'texts': result.get('texts', []),
                'panels': result.get('panels', []),
            }

    write_json(args.output, detections)


if __name__ == '__main__':
    parsed_args = parse_args()
    parsed_args.func(parsed_args)
