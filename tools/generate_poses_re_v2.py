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

    model_1 = LiftedGAN()
    model_1.load_model(args.model1)

    model_2 = LiftedGAN()
    model_2.load_model(args.model2)

    model_3 = LiftedGAN()
    model_3.load_model(args.model3)

    model_4 = LiftedGAN()
    model_4.load_model(args.model4)

    model_5 = LiftedGAN()
    model_5.load_model(args.model5)

    print('Forwarding the network...')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for head in tqdm(range(0, args.n_samples, args.batch_size)):
        with torch.no_grad():
            tail = min(args.n_samples, head + args.batch_size)
            b = tail - head

            latent = torch.randn((b, 512))

            # ------------model 1------------

            styles = model_1.generator.style(latent)
            styles = args.truncation * styles + (1 - args.truncation) * model_1.w_mu

            generate_save_image(args, b, head, model_1, styles, "_re")
            light_calculate(args, head, model_1, styles, "_lights_re")

            # ------------model 2------------

            styles = model_2.generator.style(latent)
            styles = args.truncation * styles + (1 - args.truncation) * model_2.w_mu

            generate_save_image(args, b, head, model_2, styles, "_re_wo_flip")
            light_calculate(args, head, model_2, styles, "_lights_re_wo_flip")

            # ------------model 3------------

            styles = model_3.generator.style(latent)
            styles = args.truncation * styles + (1 - args.truncation) * model_3.w_mu

            generate_save_image(args, b, head, model_3, styles, "_re_wo_idt")
            light_calculate(args, head, model_3, styles, "_lights_re_wo_idt")

            # ------------model 4------------

            styles = model_4.generator.style(latent)
            styles = args.truncation * styles + (1 - args.truncation) * model_4.w_mu

            generate_save_image(args, b, head, model_4, styles, "_re_wo_perturb")
            light_calculate(args, head, model_4, styles, "_lights_re_wo_perturb")

            # ------------model 5------------

            styles = model_5.generator.style(latent)
            styles = args.truncation * styles + (1 - args.truncation) * model_5.w_mu

            generate_save_image(args, b, head, model_5, styles, "_re_wo_rega")
            light_calculate(args, head, model_5, styles, "_lights_re_wo_rega")


def generate_save_image(args, b, head, model, styles, label):
    canon_depth, canon_albedo, canon_light, view, neutral_style, trans_map, canon_im_raw = model.estimate(
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
            angle_ = -1 * angle / model.xyz_rotation_range
            view_rotate[:, 0] = torch.ones(b) * 0.0
            view_rotate[:, 1] = torch.ones(b) * angle_
        else:
            angle_ = angle / model.xyz_rotation_range
            view_rotate[:, 0] = torch.ones(b) * angle_
            view_rotate[:, 1] = torch.ones(b) * 0
        view_rotate[:, 2] = torch.ones(b) * 0
        view_rotate[:, 3] = torch.sin(view_rotate[:, 1]) * 0.1
        view_rotate[:, 4] = - torch.sin(view_rotate[:, 0]) * 0.2
        view_rotate[:, 5] = torch.ones(b) * 0
        rocon_rotate_ = \
            model.render(canon_depth, canon_albedo, canon_light, view_rotate, trans_map=trans_map)[0]
        recon_rotate.append(rocon_rotate_.cpu())
    outputs = torch.stack(recon_rotate, 1).clamp(min=-1., max=1.)  # N x M x C x H x W
    outputs = outputs.permute(0, 1, 3, 4, 2).numpy() * 0.5 + 0.5
    outputs = (outputs * 255).astype(np.uint8)
    for i in range(outputs.shape[0]):
        mimwrite(f'{args.output_dir}/{head + i + 1}' + '_' + str(label) + '.gif', outputs[i])


def light_calculate(args, head, model, styles, label):
    canon_depth, canon_albedo, canon_light, view, neutral_style, trans_map, canon_im_raw = model.estimate(
        styles)
    recon_relight = []
    for angle in range(-60, 61, 6):
        light = canon_light.clone()
        angle_ = angle * math.pi / 180
        light[:, 0] = 0.2
        light[:, 1] = 0.8
        light[:, 2] = math.tan(angle_)
        light[:, 3] = 0
        recon_relight_ = model.render(canon_depth, canon_albedo, light, view, trans_map=trans_map)[0]
        recon_relight.append(recon_relight_.cpu())
    outputs = torch.stack(recon_relight, 1).clamp(min=-1., max=1.)  # N x M x C x H x W
    outputs = outputs.permute(0, 1, 3, 4, 2).numpy() * 0.5 + 0.5
    outputs = (outputs * 255).astype(np.uint8)
    for i in range(outputs.shape[0]):
        mimwrite(f'{args.output_dir}/{head + i + 1}' + '_' + str(label) + '.gif', outputs[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model1", help="The path to the pre-trained original model weights",
                        type=str)
    parser.add_argument("--model2", help="The path to the pre-trained reproduced model weights",
                        type=str)
    parser.add_argument("--model3", help="The path to the pre-trained original model weights",
                        type=str)
    parser.add_argument("--model4", help="The path to the pre-trained reproduced model weights",
                        type=str)
    parser.add_argument("--model5", help="The path to the pre-trained original model weights",
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
