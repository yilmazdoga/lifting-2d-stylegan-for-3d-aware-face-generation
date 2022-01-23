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

from imageio import mimwrite

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
            styles = model_original.generator.style(latent)
            styles = args.truncation * styles + (1 - args.truncation) * model_original.w_mu

            canon_depth, canon_albedo, canon_light, view, neutral_style, trans_map, canon_im_raw = model_original.estimate(
                styles)

            recon_rotate = []
            if args.type == 'yaw':
                angles = range(-45, 46, 3)
            elif args.type == 'pitch':
                angles = range(-15, 16, 1)
            else:
                raise ValueError(f'Unkown angle type: {args.type}')

            for angle in angles:
                view_rotate = view.clone()
                if args.type == 'yaw':
                    angle_ = -1 * angle / model_original.xyz_rotation_range
                    view_rotate[:, 0] = torch.ones(b) * 0.0
                    view_rotate[:, 1] = torch.ones(b) * angle_
                else:
                    angle_ = angle / model_original.xyz_rotation_range
                    view_rotate[:, 0] = torch.ones(b) * angle_
                    view_rotate[:, 1] = torch.ones(b) * 0
                view_rotate[:, 2] = torch.ones(b) * 0
                view_rotate[:, 3] = torch.sin(view_rotate[:, 1]) * 0.1
                view_rotate[:, 4] = - torch.sin(view_rotate[:, 0]) * 0.2
                view_rotate[:, 5] = torch.ones(b) * 0
                rocon_rotate_ = \
                    model_original.render(canon_depth, canon_albedo, canon_light, view_rotate, trans_map=trans_map)[0]
                recon_rotate.append(rocon_rotate_.cpu())

            outputs_original = torch.stack(recon_rotate, 1).clamp(min=-1., max=1.)  # N x M x C x H x W
            outputs_original = outputs_original.permute(0, 1, 3, 4, 2).numpy() * 0.5 + 0.5
            outputs_original = (outputs_original * 255).astype(np.uint8)

            for i in range(outputs_original.shape[0]):
                mimwrite(f'{args.output_dir}/{head + i + 1}.gif', outputs_original[i])

            # ------------Reproduced Weights------------

            styles = model_reproduced.generator.style(latent)
            styles = args.truncation * styles + (1 - args.truncation) * model_reproduced.w_mu

            canon_depth, canon_albedo, canon_light, view, neutral_style, trans_map, canon_im_raw = model_reproduced.estimate(
                styles)

            recon_rotate = []
            if args.type == 'yaw':
                angles = range(-45, 46, 3)
            elif args.type == 'pitch':
                angles = range(-15, 16, 1)
            else:
                raise ValueError(f'Unkown angle type: {args.type}')

            for angle in angles:
                view_rotate = view.clone()
                if args.type == 'yaw':
                    angle_ = -1 * angle / model_reproduced.xyz_rotation_range
                    view_rotate[:, 0] = torch.ones(b) * 0.0
                    view_rotate[:, 1] = torch.ones(b) * angle_
                else:
                    angle_ = angle / model_reproduced.xyz_rotation_range
                    view_rotate[:, 0] = torch.ones(b) * angle_
                    view_rotate[:, 1] = torch.ones(b) * 0
                view_rotate[:, 2] = torch.ones(b) * 0
                view_rotate[:, 3] = torch.sin(view_rotate[:, 1]) * 0.1
                view_rotate[:, 4] = - torch.sin(view_rotate[:, 0]) * 0.2
                view_rotate[:, 5] = torch.ones(b) * 0
                rocon_rotate_ = \
                model_reproduced.render(canon_depth, canon_albedo, canon_light, view_rotate, trans_map=trans_map)[0]
                recon_rotate.append(rocon_rotate_.cpu())

            outputs_reproduced = torch.stack(recon_rotate, 1).clamp(min=-1., max=1.)  # N x M x C x H x W
            outputs_reproduced = outputs_reproduced.permute(0, 1, 3, 4, 2).numpy() * 0.5 + 0.5
            outputs_reproduced = (outputs_reproduced * 255).astype(np.uint8)

            for i in range(outputs_reproduced.shape[0]):
                mimwrite(f'{args.output_dir}/{head + i + 1}' + '_RE.gif', outputs_reproduced[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_original", help="The path to the pre-trained original model weights",
                        type=str)

    parser.add_argument("--model_reproduced", help="The path to the pre-trained reproduced model weights",
                        type=str)
    parser.add_argument("--output_dir", help="The output path",
                        type=str)
    parser.add_argument("--type", help="The path to aligned face images",
                        type=str, default="yaw")
    parser.add_argument("--truncation", help="Truncation of latent styles",
                        type=int, default=0.7)
    parser.add_argument("--n_samples", help="Number of images to generate",
                        type=int, default=100)
    parser.add_argument("--batch_size", help="Number of images per mini batch",
                        type=int, default=16)
    args = parser.parse_args()
    main(args)
