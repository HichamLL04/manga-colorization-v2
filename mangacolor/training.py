"""Supervised fine-tuning with explicit local and global color conditioning."""
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .checkpoints import load_model, save_checkpoint
from .data import PaletteDataset


def color_losses(prediction, guide, batch):
    target, mask = batch["target"], batch["hint_mask"]
    per_image = (prediction - target).abs().mean((1, 2, 3))
    guide_target = F.interpolate(target, size=guide.shape[-2:], mode="area")
    guide_loss = (guide - guide_target).abs().mean((1, 2, 3))
    # Normalize each image by hinted pixels, not total image area.
    hint_loss = ((prediction - target).abs() * mask).sum((1, 2, 3)) / (3 * mask.sum((1, 2, 3))).clamp_min(1)
    # A soft palette preference in chromatic components; luminance remains free.
    reduced = F.adaptive_avg_pool2d(prediction, (16, 16)).permute(0, 2, 3, 1).flatten(1, 2)
    chroma = reduced - reduced.mean(-1, keepdim=True)
    palette = batch["palette"]
    palette = palette - palette.mean(-1, keepdim=True)
    distance = (chroma[:, :, None] - palette[:, None]).square().mean(-1)
    valid = batch["palette_mask"].bool()
    distance = distance.masked_fill(~valid[:, None], 1e4)
    palette_loss = distance.min(-1).values.mean(-1) * valid.any(-1)
    # Prefer real labels over unreviewed teacher outputs.
    weight = batch.get("weight", torch.ones_like(per_image)).reshape(-1)
    reduce = lambda value: (value * weight).mean()
    parts = {"reconstruction": reduce(per_image), "guide": reduce(guide_loss),
             "hint": reduce(hint_loss), "palette": reduce(palette_loss)}
    parts["total"] = parts["reconstruction"] + .3 * parts["guide"] + 2 * parts["hint"] + .1 * parts["palette"]
    return parts


def move_batch(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


@torch.inference_mode()
def validate(model, loader, device, max_batches=None):
    model.eval()
    sums, images = {}, 0
    hint_error, hint_pixels = 0., 0.
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        batch = move_batch(batch, device)
        pred, guide = model(batch["input"], batch["palette"], batch["palette_mask"])
        parts = color_losses(pred, guide, batch)
        count = pred.shape[0]
        for name, value in parts.items():
            sums[name] = sums.get(name, 0) + float(value) * count
        mask = batch["hint_mask"]
        hint_error += float(((pred - batch["target"]).abs() * mask).sum())
        hint_pixels += float(mask.sum()) * 3
        images += count
    return {**{name: value / max(1, images) for name, value in sums.items()},
            "images": images, "hint_mae_rgb_0_255": hint_error / hint_pixels * 127.5 if hint_pixels else None}


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or (args.max_steps is not None and args.max_steps < 1):
        raise ValueError("epochs, batch-size y max-steps deben ser positivos.")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("PyTorch no tiene CUDA disponible.")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "last.pt").exists() and not args.resume:
        raise FileExistsError("El directorio ya tiene un entrenamiento. Usa otro destino o --resume.")
    model, metadata = load_model(args.resume or args.checkpoint, device, allow_untrained_palette=True)
    model.allow_untrained_palette = True
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("palette_adapter.") if args.scope == "adapter" else not name.startswith("encoder."))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    start_epoch, step = 0, 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        if state.get("optimizer") is None:
            raise ValueError("El checkpoint no guarda optimizador para reanudar.")
        if state.get("metadata", {}).get("scope", args.scope) != args.scope:
            raise ValueError("Reanuda con el mismo --scope del entrenamiento original.")
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, step = state.get("epoch", 0), state.get("step", 0)
    dataset = PaletteDataset(args.manifest, split="train", size=args.size, seed=args.seed)
    if not len(dataset):
        raise ValueError("El manifiesto no tiene objetivos de entrenamiento; BW sin etiquetas no basta.")
    validation = PaletteDataset(args.manifest, split="val", size=args.size, seed=args.seed)
    val_loader = DataLoader(validation, batch_size=args.batch_size, num_workers=0)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    metadata = {**metadata, "scope": args.scope, "size": args.size, "seed": args.seed,
                "manifest": str(Path(args.manifest).resolve()), "training_status": "experimental",
                "quality_validated": False, "loss": "l1+guide+masked_hint+chroma_palette",
                "train_samples": len(dataset), "validation_samples": len(validation)}
    configuration = vars(args).copy()
    configuration.pop("func", None)
    (output / "config.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    updates = 0
    palette_updates = int(metadata.get("palette_updates", 0))
    log = output / "metrics.jsonl"
    for epoch in range(start_epoch, start_epoch + args.epochs):
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        generator = torch.Generator().manual_seed(args.seed + epoch)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, generator=generator)
        model.train()
        model.encoder.eval()  # Preserve pretrained statistics on small manga batches.
        if args.scope == "adapter":
            model.eval()
            model.palette_adapter.train()
        for batch in loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                pred, guide = model(batch["input"], batch["palette"], batch["palette_mask"])
                losses = color_losses(pred, guide, batch)
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError("Pérdida no finita; no se guarda un checkpoint defectuoso.")
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
            palette_gradient = any(p.grad is not None and bool(p.grad.abs().sum() > 0)
                                   for p in model.palette_adapter.parameters())
            scaler.step(optimizer)
            scaler.update()
            step += 1
            updates += 1
            if bool(batch["palette_mask"].any()) and palette_gradient:
                palette_updates += 1
            event = {"epoch": epoch, "step": step, **{name: float(value.detach()) for name, value in losses.items()}}
            print(json.dumps(event), flush=True)
            with log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event) + "\n")
            if args.max_steps and updates >= args.max_steps:
                break
        metrics = validate(model, val_loader, device, args.validation_batches) if len(validation) else {"images": 0}
        metadata.update(palette_trained=palette_updates > 0,
                        palette_updates=palette_updates, validation=metrics)
        # Epoch-level resume intentionally starts a fresh shuffled epoch.
        save_checkpoint(output / "last.pt", model, metadata, optimizer, epoch=epoch + 1, step=step)
        with log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"epoch": epoch, "validation": metrics}) + "\n")
        print(json.dumps({"checkpoint": str(output / "last.pt"), "validation": metrics}), flush=True)
        if args.max_steps and updates >= args.max_steps:
            break
    return output / "last.pt"
