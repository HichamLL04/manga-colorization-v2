import argparse
import json
from pathlib import Path


def parse_export_size(value):
    if not value or str(value).lower() == "original":
        return "original"
    val_str = str(value).lower()
    if "x" in val_str:
        parts = val_str.split("x")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return int(parts[0]), int(parts[1])
    try:
        return int(val_str)
    except ValueError:
        raise ValueError(f"Formato de export-size no reconocido: '{value}'. Usa 'original', un entero o WxH (ej. 1125x1600).")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Colorización v2 con pistas regionales, paleta y entrenamiento")
    parser.add_argument("--threads", type=int, default=4, help="Hilos CPU de PyTorch")
    commands = parser.add_subparsers(dest="command", required=True)
    color = commands.add_parser("colorize", help="Colorear una imagen con pistas PNG RGBA y/o paleta")
    color.add_argument("input")
    color.add_argument("output")
    color.add_argument("--checkpoint", default="v2generator.zip")
    color.add_argument("--hints", help="PNG RGBA del tamaño de la imagen; alfa = máscara")
    color.add_argument("--points", nargs="*", default=[], help="Puntos de color 'x,y:#RRGGBB' o 'x,y,radio:#RRGGBB'")
    color.add_argument("--palette", nargs="*", default=[])
    color.add_argument("--palette-mode", choices=["hints", "learned"], default="hints")
    color.add_argument("--lock-hints", action="store_true", help="Fijar exactamente las zonas pintadas tras inferencia")
    color.add_argument("--preserve-ink", action="store_true")
    color.add_argument("--size", type=int, default=576, help="Tamaño canónico de inferencia")
    color.add_argument("--device", default="cpu")
    color.add_argument("--export-size", default="original", help="Tamaño de exportación: 'original', número o WxH")
    color.add_argument("--reconstruct-mode", choices=["guided", "legacy"], default="guided",
                       help="Modo de reconstrucción: guided (filtrado guiado Lab desacoplado) o legacy")
    color.add_argument("--save-color-field", help="Ruta .npz para guardar el campo canónico predicho")
    color.add_argument("--reuse-color-field", help="Ruta .npz para reutilizar un campo canónico sin inferir de nuevo")
    color.add_argument("--text-mask", help="PNG de máscara binaria para proteger texto con 0 error")
    color.add_argument("--ink-mask", help="PNG de máscara binaria para proteger tinta")
    color.add_argument("--detail-strength", type=float, default=1.0, help="Fuerza del detalle de luminosidad original")
    color.add_argument("--alpha-original", type=float, default=0.0, help="Mezcla con luminosidad original (0.0=predicha, 1.0=original)")
    color.add_argument("--overwrite", action="store_true", help="Permitir sobrescribir destino existente")

    recon = commands.add_parser("reconstruct", help="Reconstruir a alta resolución desde un campo canónico .npz sin red neuronal")
    recon.add_argument("input", help="Imagen original (RGB o BW)")
    recon.add_argument("color_field", help="Archivo .npz con el campo canónico predicho")
    recon.add_argument("output", help="Ruta de exportación (.png)")
    recon.add_argument("--export-size", default="original", help="Tamaño de exportación: 'original', entero o WxH")
    recon.add_argument("--text-mask", help="PNG de máscara binaria para proteger texto")
    recon.add_argument("--ink-mask", help="PNG de máscara binaria para proteger tinta")
    recon.add_argument("--hints-lock", help="PNG RGBA para bloqueo exacto de pistas")
    recon.add_argument("--lock-hints", action="store_true", help="Activar bloqueo exacto de pistas")
    recon.add_argument("--detail-strength", type=float, default=1.0)
    recon.add_argument("--alpha-original", type=float, default=0.0)
    recon.add_argument("--preserve-ink", action="store_true")
    recon.add_argument("--overwrite", action="store_true", help="Permitir sobrescribir destino existente")

    prepare = commands.add_parser("prepare", help="Crear pares sintéticos/alineados o registrar BW sin etiquetas")
    prepare.add_argument("input")
    prepare.add_argument("output")
    prepare.add_argument("--bw-dir")
    prepare.add_argument("--val-fraction", type=float, default=.1)
    prepare.add_argument("--seed", type=int, default=42)

    training = commands.add_parser("train", help="Afinar v2 y el adaptador global de paleta")
    training.add_argument("manifest")
    training.add_argument("output")
    training.add_argument("--checkpoint", default="v2generator.zip")
    training.add_argument("--resume")
    training.add_argument("--device", default="cpu")
    training.add_argument("--size", type=int, default=256)
    training.add_argument("--epochs", type=int, default=10)
    training.add_argument("--batch-size", type=int, default=1)
    training.add_argument("--lr", type=float, default=2e-5)
    training.add_argument("--max-steps", type=int)
    training.add_argument("--scope", choices=["decoder", "adapter"], default="decoder")
    training.add_argument("--workers", type=int, default=0)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--validation-batches", type=int, default=None)

    export = commands.add_parser("export-v2", help="Exportar pesos nativos de 5 canales; no incluye el adaptador de paleta")
    export.add_argument("checkpoint")
    export.add_argument("output")

    pseudo = commands.add_parser("pseudo-label", help="Crear pseudoetiquetas de BW, claramente marcadas")
    pseudo.add_argument("manifest")
    pseudo.add_argument("output")
    pseudo.add_argument("--checkpoint", default="v2generator.zip")
    pseudo.add_argument("--device", default="cpu")
    pseudo.add_argument("--size", type=int, default=576)

    editor = commands.add_parser("edit", help="Editor local de pincel y paleta en el navegador")
    editor.add_argument("--checkpoint", default="v2generator.zip")
    editor.add_argument("--device", default="cpu")
    editor.add_argument("--port", type=int, default=7860)

    inspect = commands.add_parser("inspect", help="Validar checkpoint y mostrar capacidades")
    inspect.add_argument("checkpoint", nargs="?", default="v2generator.zip")
    args = parser.parse_args(argv)
    import torch
    if args.threads < 1:
        parser.error("threads debe ser positivo")
    torch.set_num_threads(args.threads)
    try:
        if args.command == "prepare":
            from .data import prepare_dataset
            print(prepare_dataset(args.input, args.output, args.bw_dir, args.val_fraction, args.seed))
        elif args.command == "train":
            from .training import train
            train(args)
        elif args.command == "colorize":
            from PIL import Image
            from .inference import ColorizationPipeline, open_rgb, parse_palette
            from .reconstruction import (
                ColorField,
                Conditions,
                ExportConfig,
                InferenceConfig,
                Masks,
                ReconstructionConfig,
                export_result,
                predict_color_field,
                reconstruct,
            )
            destination = Path(args.output)
            if destination.exists() and not getattr(args, "overwrite", False):
                raise FileExistsError(f"Ya existe {destination}; OUT01 prohíbe sobrescribir sin permiso.")

            input_img = open_rgb(args.input)
            hints_img = Image.open(args.hints).convert("RGBA") if args.hints else None
            if args.points:
                from PIL import ImageDraw
                if hints_img is None:
                    hints_img = Image.new("RGBA", input_img.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(hints_img)
                for pt in args.points:
                    coords, hex_col = pt.split(":")
                    parts = [int(v.strip()) for v in coords.split(",")]
                    px, py = parts[0], parts[1]
                    radius = parts[2] if len(parts) > 2 else 8
                    clean = hex_col.strip().lstrip("#")
                    cr, cg, cb = int(clean[0:2], 16), int(clean[2:4], 16), int(clean[4:6], 16)
                    draw.ellipse([px - radius, py - radius, px + radius, py + radius], fill=(cr, cg, cb, 255))
            palette_vals = parse_palette(args.palette)
            export_dim = parse_export_size(args.export_size)

            # Reusing existing canonical color field (AT07, RES01, RES02)
            if args.reuse_color_field:
                cf = ColorField.load_npz(args.reuse_color_field)
            else:
                pipeline = ColorizationPipeline(args.checkpoint, args.device)
                if args.reconstruct_mode == "legacy" and export_dim == "original" and not args.text_mask and not args.ink_mask and not args.save_color_field:
                    result, report = pipeline.colorize(
                        input_img, hints_img, palette_vals,
                        args.size, args.palette_mode, args.lock_hints, args.preserve_ink,
                    )
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    result.save(destination)
                    destination.with_suffix(destination.suffix + ".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                    print(destination.resolve())
                    return
                conditions = Conditions(hints=hints_img, palette=palette_vals, palette_mode=args.palette_mode)
                inf_cfg = InferenceConfig(canonical_size=args.size, device=args.device)
                cf = predict_color_field(input_img, conditions, pipeline.model, inf_cfg, pipeline.metadata)
                if args.save_color_field:
                    cf.save_npz(args.save_color_field)

            text_mask_img = Image.open(args.text_mask) if args.text_mask else None
            ink_mask_img = Image.open(args.ink_mask) if args.ink_mask else None
            masks = Masks(text_mask=text_mask_img, ink_mask=ink_mask_img, hints_lock=hints_img, lock_hints=args.lock_hints)
            recon_cfg = ReconstructionConfig(
                export_size=export_dim,
                detail_strength=args.detail_strength,
                alpha_original=args.alpha_original,
                preserve_ink=args.preserve_ink,
            )
            render_res = reconstruct(input_img, cf, masks=masks, reconstruction_config=recon_cfg)
            exp_cfg = ExportConfig(overwrite=getattr(args, "overwrite", False), save_report=True)
            exp_rec = export_result(render_res, destination, exp_cfg)
            print(exp_rec.path.resolve())

        elif args.command == "reconstruct":
            from PIL import Image
            from .inference import open_rgb
            from .reconstruction import (
                ColorField,
                ExportConfig,
                Masks,
                ReconstructionConfig,
                export_result,
                reconstruct,
            )
            destination = Path(args.output)
            if destination.exists() and not getattr(args, "overwrite", False):
                raise FileExistsError(f"Ya existe {destination}; OUT01 prohíbe sobrescribir sin permiso.")

            input_img = open_rgb(args.input)
            cf = ColorField.load_npz(args.color_field)
            export_dim = parse_export_size(args.export_size)
            text_mask_img = Image.open(args.text_mask) if args.text_mask else None
            ink_mask_img = Image.open(args.ink_mask) if args.ink_mask else None
            hints_lock_img = Image.open(args.hints_lock).convert("RGBA") if args.hints_lock else None

            masks = Masks(
                text_mask=text_mask_img,
                ink_mask=ink_mask_img,
                hints_lock=hints_lock_img,
                lock_hints=args.lock_hints or (hints_lock_img is not None),
            )
            recon_cfg = ReconstructionConfig(
                export_size=export_dim,
                detail_strength=args.detail_strength,
                alpha_original=args.alpha_original,
                preserve_ink=args.preserve_ink,
            )
            render_res = reconstruct(input_img, cf, masks=masks, reconstruction_config=recon_cfg)
            exp_cfg = ExportConfig(overwrite=getattr(args, "overwrite", False), save_report=True)
            exp_rec = export_result(render_res, destination, exp_cfg)
            print(exp_rec.path.resolve())
        elif args.command == "pseudo-label":
            from .inference import ColorizationPipeline, pseudo_label
            print(pseudo_label(args.manifest, args.output, ColorizationPipeline(args.checkpoint, args.device), args.size))
        elif args.command == "export-v2":
            from .checkpoints import load_model, export_native
            if Path(args.output).exists():
                raise FileExistsError(args.output)
            model, _ = load_model(args.checkpoint)
            export_native(args.output, model)
            print("Exportado v2 nativo: admite pistas regionales; la paleta global aprendida requiere el checkpoint completo.")
        elif args.command == "edit":
            from .editor import serve
            serve(args.checkpoint, args.device, args.port)
        elif args.command == "inspect":
            from .checkpoints import load_model
            _, metadata = load_model(args.checkpoint)
            print(json.dumps(metadata, indent=2, ensure_ascii=False))
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
