import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import os
import sys
import time
import math
import argparse
import numpy as np
from tqdm import tqdm
import torch

from imageio import imwrite

import utils
from models.lifted_gan import LiftedGAN


def main(args):
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    model_original = LiftedGAN()
    model_original.load_model(args.model_original)

    model_reproduced = LiftedGAN()
    model_reproduced.load_model(args.model_reproduced)

    print('Forwarding the network...')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for head in tqdm(range(0, args.n_samples, args.batch_size)):
        with torch.no_grad():
            tail = min(args.n_samples, head + args.batch_size)
            b = tail - head

            latent = torch.randn((b, 512))

            # ------------Original Weights------------
            styles_original = model_original.generator.style(latent)
            styles_original = args.truncation * styles_original + (1 - args.truncation) * model_original.w_mu

            canon_depth, canon_albedo, canon_light, view, neutral_style, trans_map, canon_im_raw = model_original.estimate(
                styles_original)

            recon_im_original = model_original.render(canon_depth, canon_albedo, canon_light, view, trans_map=trans_map)[0]

            outputs_original = recon_im_original.permute(0, 2, 3, 1).cpu().numpy() * 0.5 + 0.5
            outputs_original = np.minimum(1.0, np.maximum(0.0, outputs_original))
            outputs_original = (outputs_original * 255).astype(np.uint8)

            for i in range(outputs_original.shape[0]):
                imwrite(f'{args.output_dir}/{head + i + 1:05d}.png', outputs_original[i])

            # ------------Reproduced Weights------------
            styles_reproduced = model_reproduced.generator.style(latent)
            styles_reproduced = args.truncation * styles_reproduced + (1 - args.truncation) * model_reproduced.w_mu

            canon_depth, canon_albedo, canon_light, view, neutral_style, trans_map, canon_im_raw = model_reproduced.estimate(
                styles_reproduced)

            recon_im_reproduced = model_reproduced.render(canon_depth, canon_albedo, canon_light, view, trans_map=trans_map)[0]

            outputs_reproduced = recon_im_reproduced.permute(0, 2, 3, 1).cpu().numpy() * 0.5 + 0.5
            outputs_reproduced = np.minimum(1.0, np.maximum(0.0, outputs_reproduced))
            outputs_reproduced = (outputs_reproduced * 255).astype(np.uint8)

            tag = 'RE_'
            for i in range(outputs_reproduced.shape[0]):
                imwrite(f'{args.output_dir}/{tag + head + i + 1:05d}.png', outputs_reproduced[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_original", help="The path to the pre-trained original model weights",
                        type=str)

    parser.add_argument("--model_reproduced", help="The path to the pre-trained reproduced model weights",
                        type=str)

    parser.add_argument("--output_dir", help="The output path",
                        type=str)
    parser.add_argument("--truncation", help="Truncation of latent styles",
                        type=int, default=0.7)
    parser.add_argument("--n_samples", help="Number of images to generate",
                        type=int, default=50000)
    parser.add_argument("--batch_size", help="Number of images per mini batch",
                        type=int, default=16)
    args = parser.parse_args()
    main(args)
