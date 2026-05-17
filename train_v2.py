import argparse
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def load_optional_state(module, path, strict=True):
    if not path:
        return
    if not Path(path).exists():
        print('Warning: weights not found, skipping: {}'.format(path))
        return
    state = torch.load(path, map_location='cpu')
    module.load_state_dict(state, strict=strict)


def generator_step(batch, colorizer, discriminator, content, optimizer, args, device):
    discriminator.requires_grad_(False)
    colorizer.generator.requires_grad_(True)
    optimizer.zero_grad(set_to_none=True)

    sketch = batch['sketch'].to(device)
    color = batch['color'].to(device)
    fake, guide = colorizer(sketch)
    logits_fake = discriminator(fake)
    real_labels = torch.ones((sketch.size(0), 1), device=device)
    adv_loss = nn.functional.binary_cross_entropy_with_logits(logits_fake, real_labels)
    l1_loss = nn.functional.l1_loss(fake, color)
    aux_loss = nn.functional.l1_loss(guide, color)

    if args.content_weight > 0:
        content_fake = content(fake)
        with torch.no_grad():
            content_true = content(color)
        perceptual_loss = nn.functional.mse_loss(content_fake, content_true)
    else:
        perceptual_loss = torch.zeros((), device=device)

    loss = (
        args.l1_weight * (l1_loss + args.aux_weight * aux_loss)
        + args.adv_weight * adv_loss
        + args.content_weight * perceptual_loss
    )

    if args.white_weight > 0:
        loss = loss + args.white_weight * white_color_penalty(fake, color)

    loss.backward()
    optimizer.step()
    return {
        'g_loss': float(loss.detach().cpu()),
        'g_l1': float(l1_loss.detach().cpu()),
        'g_adv': float(adv_loss.detach().cpu()),
    }


def discriminator_step(batch, colorizer, discriminator, optimizer, device):
    discriminator.requires_grad_(True)
    colorizer.generator.requires_grad_(False)
    optimizer.zero_grad(set_to_none=True)

    sketch = batch['sketch'].to(device)
    color = batch['color'].to(device)
    real_labels = torch.full((sketch.size(0), 1), 0.9, device=device)
    fake_labels = torch.zeros((sketch.size(0), 1), device=device)

    with torch.no_grad():
        fake, _ = colorizer(sketch)

    logits_real = discriminator(color)
    logits_fake = discriminator(fake.detach())
    real_loss = nn.functional.binary_cross_entropy_with_logits(logits_real, real_labels)
    fake_loss = nn.functional.binary_cross_entropy_with_logits(logits_fake, fake_labels)
    loss = real_loss + fake_loss
    loss.backward()
    optimizer.step()
    return {'d_loss': float(loss.detach().cpu())}


def save_checkpoint(output_dir, epoch, colorizer, discriminator, opt_g, opt_d, args):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'epoch': epoch,
        'generator': colorizer.generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'optimizer_g': opt_g.state_dict(),
        'optimizer_d': opt_d.state_dict(),
        'args': vars(args),
    }
    torch.save(payload, output_dir / 'checkpoint_latest.pt')
    torch.save(colorizer.generator.state_dict(), output_dir / 'generator_latest.pt')
    torch.save(colorizer.generator.state_dict(), output_dir / 'generator_epoch_{:03d}.pt'.format(epoch))


def train(args):
    global torch, nn, optim, DataLoader
    global Colorizer, V2AlignedPanelDataset, Discriminator, VGGContentLoss
    global weights_init, weights_init_spectral, white_color_penalty

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    from networks.models import Colorizer
    from training_v2.data import V2AlignedPanelDataset
    from training_v2.losses import (
        Discriminator,
        VGGContentLoss,
        weights_init,
        weights_init_spectral,
        white_color_penalty,
    )

    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    dataset = V2AlignedPanelDataset(
        args.dataset,
        split=args.split,
        crop_size=args.crop_size,
        input_channels=args.input_channels,
        augment=not args.no_augment,
        empty_hint_prob=args.empty_hint_prob,
        full_hint_prob=args.full_hint_prob,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device == 'cuda',
        drop_last=True,
    )

    colorizer = Colorizer(input_channels=args.input_channels).to(device)
    discriminator = Discriminator().to(device)
    content = VGGContentLoss(args.vgg_weights).eval().to(device) if args.content_weight > 0 else None
    colorizer.generator.apply(weights_init)
    discriminator.apply(weights_init_spectral)
    if args.extractor_weights:
        load_optional_state(colorizer.generator.encoder, args.extractor_weights, strict=False)
    load_optional_state(colorizer.generator, args.resume_generator, strict=True)
    freeze_encoder = args.freeze_encoder or bool(args.extractor_weights)
    if freeze_encoder:
        colorizer.generator.encoder.eval()
        for param in colorizer.generator.encoder.parameters():
            param.requires_grad = False

    opt_g = optim.Adam(
        [param for param in colorizer.generator.parameters() if param.requires_grad],
        lr=args.generator_lr,
        betas=(0.5, 0.9),
    )
    opt_d = optim.Adam(discriminator.parameters(), lr=args.discriminator_lr, betas=(0.5, 0.9))

    step_generator = False
    for epoch in range(1, args.epochs + 1):
        if epoch == args.lr_decay_epoch:
            for group in opt_g.param_groups:
                group['lr'] *= 0.1
            for group in opt_d.param_groups:
                group['lr'] *= 0.1

        colorizer.generator.train()
        if freeze_encoder:
            colorizer.generator.encoder.eval()
        discriminator.train()
        iterator = tqdm(dataloader, desc='Epoch {}/{}'.format(epoch, args.epochs), unit='batch')
        totals = {'g_loss': 0.0, 'd_loss': 0.0}
        counts = {'g_loss': 0, 'd_loss': 0}

        for batch in iterator:
            if step_generator:
                metrics = generator_step(batch, colorizer, discriminator, content, opt_g, args, device)
            else:
                metrics = discriminator_step(batch, colorizer, discriminator, opt_d, device)
            step_generator = not step_generator

            for key, value in metrics.items():
                if key in totals:
                    totals[key] += value
                    counts[key] += 1
            if hasattr(iterator, 'set_postfix'):
                iterator.set_postfix({key: totals[key] / max(1, counts[key]) for key in totals})

        save_checkpoint(args.output_dir, epoch, colorizer, discriminator, opt_g, opt_d, args)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a manga-colorization-v2 compatible generator.')
    parser.add_argument('--dataset', required=True, help='Dataset root produced by build_dataset.py.')
    parser.add_argument('--output-dir', default='runs/v2_training')
    parser.add_argument('--split', default='train')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--crop-size', type=int, default=512)
    parser.add_argument('--input-channels', type=int, choices=(5, 6), default=5)
    parser.add_argument('--generator-lr', type=float, default=1e-4)
    parser.add_argument('--discriminator-lr', type=float, default=4e-4)
    parser.add_argument('--lr-decay-epoch', type=int, default=10)
    parser.add_argument('--l1-weight', type=float, default=10.0)
    parser.add_argument('--aux-weight', type=float, default=0.9)
    parser.add_argument('--adv-weight', type=float, default=1.0)
    parser.add_argument('--content-weight', type=float, default=0.0)
    parser.add_argument('--white-weight', type=float, default=1.0)
    parser.add_argument('--empty-hint-prob', type=float, default=0.49)
    parser.add_argument('--full-hint-prob', type=float, default=0.01)
    parser.add_argument('--no-augment', action='store_true')
    parser.add_argument('--extractor-weights')
    parser.add_argument('--freeze-encoder', action='store_true')
    parser.add_argument('--vgg-weights')
    parser.add_argument('--resume-generator')
    return parser.parse_args()


if __name__ == '__main__':
    train(parse_args())
