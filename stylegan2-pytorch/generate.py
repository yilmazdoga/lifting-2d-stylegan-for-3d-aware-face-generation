import argparse

import math
import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm
def generate(args, g_ema, device, mean_latent):

    with torch.no_grad():
        g_ema.eval()
        sample_z = torch.randn(args.sample, args.latent, device=device)
        for i in tqdm(range(args.pics)):
           truncation = args.truncation / (args.pics-1) * i
           print(truncation)
           sample, _ = g_ema([sample_z], truncation=truncation, truncation_latent=mean_latent)
           utils.save_image(
            sample,
            f'generated_samples/{str(i).zfill(6)}.png',
            nrow=int(math.sqrt(args.sample)),
            normalize=True,
            range=(-1, 1),
        )

if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--sample', type=int, default=1)
    parser.add_argument('--pics', type=int, default=20)
    parser.add_argument('--truncation', type=float, default=1)
    parser.add_argument('--truncation_mean', type=int, default=4096)
    parser.add_argument('--ckpt', type=str, default="stylegan2-ffhq-config-f.pt")
    parser.add_argument('--channel_multiplier', type=int, default=2)

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt)

    g_ema.load_state_dict(checkpoint['g_ema'], strict=False)

    if True or args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    generate(args, g_ema, device, mean_latent)
