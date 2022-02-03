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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LiftedGAN()
    model.load_model(args.model)

    print('Forwarding the network...')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for head in tqdm(range(0, args.n_samples, args.batch_size)):
        with torch.no_grad():
            tail = min(args.n_samples, head + args.batch_size)
            b = tail - head

            latent1 = torch.randn((b, 512))
            styles1 = model.generator.style(latent1)
            styles1 = args.truncation * styles1 + (1 - args.truncation) * model.w_mu

            # tensor = torch.load(args.style)
            #
            # style_latent = tensor['test_files/mona_lisa2.jpeg']["latent"]
            # style_latent = style_latent.to(device)
            # mona_lisa = args.truncation * style_latent + (1 - args.truncation) * model.w_mu

            latent2 = torch.randn((b, 512))
            styles2 = model.generator.style(latent2)
            styles2 = args.truncation * styles2 + (1 - args.truncation) * model.w_mu

            canon_depth, canon_albedo, canon_light, view, neutral_style, trans_map, canon_im_raw = model.estimate(styles1)
            render_save(args, canon_albedo, canon_depth, canon_light, head, model, trans_map, view, "orijinal")

            canon_depth, canon_albedo, canon_light, view, neutral_style, trans_map, canon_im_raw = model.estimate(styles2)
            render_save(args, canon_albedo, canon_depth, canon_light, head, model, trans_map, view, "style")

            canon_depth, canon_albedo, canon_light, view, neutral_style, trans_map, canon_im_raw = model.estimate_w_style(styles1, styles2)
            render_save(args, canon_albedo, canon_depth, canon_light, head, model, trans_map, view, "orijinal_w_style")


def render_save(args, canon_albedo, canon_depth, canon_light, head, model, trans_map, view, label):
    recon_im = model.render(canon_depth, canon_albedo, canon_light, view, trans_map=trans_map)[0]
    outputs = recon_im.permute(0, 2, 3, 1).cpu().numpy() * 0.5 + 0.5
    outputs = np.minimum(1.0, np.maximum(0.0, outputs))
    outputs = (outputs * 255).astype(np.uint8)
    for i in range(outputs.shape[0]):
        imwrite(f'{args.output_dir}/{head + i + 1:05d}' + '_' + str(label) + '.png', outputs[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="The path to the pre-trained model",
                        type=str)
    parser.add_argument("--style", help="style",
                        type=str)
    parser.add_argument("--output_dir", help="The output path",
                        type=str)
    parser.add_argument("--truncation", help="Truncation of latent styles",
                        type=int, default=0.7)
    parser.add_argument("--n_samples", help="Number of images to generate",
                        type=int, default=1)
    parser.add_argument("--batch_size", help="Number of images per mini batch",
                        type=int, default=1)
    args = parser.parse_args()
    main(args)
