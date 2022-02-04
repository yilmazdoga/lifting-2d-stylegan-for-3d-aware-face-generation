# [Re] - Lifting 2D StyleGAN for 3D-Aware Face Generation
This repository is the re-production implementation of [Lifting 2D StyleGAN for 3D-Aware Face Generation](https://arxiv.org/abs/2011.13126) by [Yichun Shi](https://seasonsh.github.io), [Divyansh Aggarwal](https://divyanshaggarwal.github.io)and Anil K. Jain in the scope of [ML Reproducibility Challenge 2021](https://paperswithcode.com/rc2021).

## Requirements
You can create the conda environment by using:
```setup
conda env create -f environment.yml
```

## Training
### Training from pre-trained StyleGAN2 (FFHQ and AFHQ Cat)
Download pre-trained StyleGAN and face embedding network from [here](https://drive.google.com/file/d/1qVoWu_fps17iTzYptwuN3ptgYeCpIl2e/view?usp=sharing) for training. Unzip them into the `pretrained/` folder. Then you can start training by:
```sh
python tools/train.py config/ffhq_256.py
```

And similarly you can start training for AFHQ Cat by:
```sh
python tools/train.py config/cats_256.py
```

### Training from pre-trained StyleGAN2 (CelebA)
In addition to instructions above download and place `checkpoint_stylegan_celeba` folder under `pretrained/`. Then you can start training by:

And similarly you can start training for AFHQ Cat by:
```sh
python tools/train.py config/celeba_256.py
```

### Training from custom data
As the original repository, we use a re-cropped version of FFHQ to fit the style of our face embedding network. You can find this dataset [here](https://drive.google.com/file/d/1pLHzbZS52XGyejubv5tT0CqhpsocaYuD/view?usp=sharing). The cats dataset can be found [here](https://drive.google.com/file/d/1soEXvvLV0uhasg9GlVhH5YW_9FsAmb3d/view?usp=sharing).
To train a StyleGAN2 from you own dataset, check the content under [`stylegan2-pytorch`](https://github.com/seasonSH/LiftedGAN/tree/main/stylegan2-pytorch) folder. After training a StyleGAN2, you can lift it using the training code.

## Testing
### Original Pre-trained LiftedGAN Models: 
[Google Drive](https://drive.google.com/file/d/1-44Eivt7GHINkX6zox89HHttujYWThz2/view?usp=sharing)

### Reproduced Pre-trained LiftedGAN and StyleGAN2 Models: 
[Google Drive](https://drive.google.com/file/d/1NE8Tfqkr4po63dMnwwnV_Q-mluYgUsLV/view?usp=sharing)

### Sampling random faces
You can generate random samples from a lifted gan by running:
```sh
python tools/generate_images.py /path/to/the/checkpoint --output_dir your/output/dir
```
Make sure the checkpoint file and its `config.py` file are under the same folder.

### Sampling random faces (using same latent vector)
You can generate random samples from 2 different LiftedGANs (whit same latent vector) by running:
```sh
python tools/generate_images_re.py --model_original /path/to/the/checkpoint --model_reproduced /path/to/the/checkpoint --output_dir your/output/dir
```

### Running Viewpoint Manipulation
You can run viewpoint manipulation for a single LiftedGAN by:
```sh
python tools/generate_poses.py /path/to/the/checkpoint --output_dir your/output/dir --type yaw
```


### Running Viewpoint Manipulation (using same latent vector)
You can run viewpoint manipulation by using 2 different LiftedGANs (whit same latent vector) by:
```sh
python tools/generate_poses_re.py --model_original /path/to/the/checkpoint --model_reproduced /path/to/the/checkpoint --output_dir your/output/dir --type yaw
```


### Running Light Direction Manipulation
You can run light direction manipulation for a single LiftedGAN by:
```sh
python tools/generate_lighting.py --model_original /path/to/the/checkpoint --model_reproduced /path/to/the/checkpoint --output_dir your/output/dir
```

### Running Light Direction Manipulation (using same latent vector)
You can run light direction manipulation by using 2 different LiftedGANs (whit same latent vector) by:
```sh
python tools/generate_lighting_re.py --model_original /path/to/the/checkpoint --model_reproduced /path/to/the/checkpoint --output_dir your/output/dir
```


### Running pose interpolation
You can run the command below to interpolate between two face poses:
```sh
python tools/generate_poses_interpolate.py /path/to/the/checkpoint --output_dir your/output/dir
```


### Running pose interpolation  (using same latent vector)
You can run the command below to interpolate between two face poses:
```sh
python tools/generate_poses_interpolate_re.py --model_original /path/to/the/checkpoint --model_reproduced /path/to/the/checkpoint --output_dir your/output/dir --type yaw
```


For all experiments make sure the checkpoint file and its `config.py` file are under the same folder. For viewpoint manipulation experiments you can change the type parameter to toggle between yaw and pitch manipulation.


### Testing FID
We use the code from rosinality's stylegan2-pytorch to compute FID. To compute the FID, you first need to compute the statistics of real images:
```sh
python utils/calc_inception.py /path/to/the/dataset/lmdb
```
You might skip this step if you are using our pre-calculated statistics file ([link](https://drive.google.com/file/d/1qVoWu_fps17iTzYptwuN3ptgYeCpIl2e/view?usp=sharing)). Then, to test the FID, you can run:
```sh
python tools/test_fid.py /path/to/the/checkpoint --inception /path/to/the/inception/file
```

## Additional Results of Our Reproduction Paper

### FFHQ Experiments

#### Face Generation
| Original | Reproduced |
|----------|------------|
| ![](readme_assets/faces/00001.png) | ![](readme_assets/faces/00001_RE.png) |
| ![](readme_assets/faces/00002.png) | ![](readme_assets/faces/00002_RE.png) |
| ![](readme_assets/faces/00003.png) | ![](readme_assets/faces/00003_RE.png) |
| ![](readme_assets/faces/00004.png) | ![](readme_assets/faces/00004_RE.png) |
| ![](readme_assets/faces/00005.png) | ![](readme_assets/faces/00005_RE.png) |
| ![](readme_assets/faces/00006.png) | ![](readme_assets/faces/00006_RE.png) |
| ![](readme_assets/faces/00007.png) | ![](readme_assets/faces/00007_RE.png) |
| ![](readme_assets/faces/00008.png) | ![](readme_assets/faces/00008_RE.png) |

### Viewpoint Manipulation (yaw)
| Original | Reproduced |
|----------|------------|
| ![](readme_assets/faces_yaw/10.gif) | ![](readme_assets/faces_yaw/10_RE.gif)|
| ![](readme_assets/faces_yaw/11.gif) | ![](readme_assets/faces_yaw/11_RE.gif)|
| ![](readme_assets/faces_yaw/12.gif) | ![](readme_assets/faces_yaw/12_RE.gif)|
| ![](readme_assets/faces_yaw/13.gif) | ![](readme_assets/faces_yaw/13_RE.gif)|
| ![](readme_assets/faces_yaw/14.gif) | ![](readme_assets/faces_yaw/14_RE.gif)|
| ![](readme_assets/faces_yaw/15.gif) | ![](readme_assets/faces_yaw/15_RE.gif)|
| ![](readme_assets/faces_yaw/16.gif) | ![](readme_assets/faces_yaw/16_RE.gif)|
| ![](readme_assets/faces_yaw/17.gif) | ![](readme_assets/faces_yaw/17_RE.gif)|

### Viewpoint Manipulation (pitch)
| Original | Reproduced |
|----------|------------|
| ![](readme_assets/faces_pitch/10.gif) | ![](readme_assets/faces_pitch/10_RE.gif) |
| ![](readme_assets/faces_pitch/11.gif) | ![](readme_assets/faces_pitch/11_RE.gif) |
| ![](readme_assets/faces_pitch/12.gif) | ![](readme_assets/faces_pitch/12_RE.gif) |
| ![](readme_assets/faces_pitch/13.gif) | ![](readme_assets/faces_pitch/13_RE.gif) |
| ![](readme_assets/faces_pitch/14.gif) | ![](readme_assets/faces_pitch/14_RE.gif) |
| ![](readme_assets/faces_pitch/15.gif) | ![](readme_assets/faces_pitch/15_RE.gif) |
| ![](readme_assets/faces_pitch/16.gif) | ![](readme_assets/faces_pitch/16_RE.gif) |
| ![](readme_assets/faces_pitch/21.gif) | ![](readme_assets/faces_pitch/21_RE.gif) |

### Re-lighting
| Original | Reproduced |
|----------|------------|
| ![](readme_assets/faces_light/1.gif) | ![](readme_assets/faces_light/1.gif) |
| ![](readme_assets/faces_light/2.gif) | ![](readme_assets/faces_light/2.gif) |
| ![](readme_assets/faces_light/3.gif) | ![](readme_assets/faces_light/3.gif) |
| ![](readme_assets/faces_light/4.gif) | ![](readme_assets/faces_light/4.gif) |
| ![](readme_assets/faces_light/5.gif) | ![](readme_assets/faces_light/5.gif) |
| ![](readme_assets/faces_light/6.gif) | ![](readme_assets/faces_light/6.gif) |
| ![](readme_assets/faces_light/7.gif) | ![](readme_assets/faces_light/7.gif) |
| ![](readme_assets/faces_light/8.gif) | ![](readme_assets/faces_light/8.gif) |

### CelebA Experiments

| Face Generation | Viewpoint Manipulation (yaw) | Viewpoint Manipulation (pitch) | Re-lighting |
|-----------------|------------------------------|--------------------------------|-------------|
|![](readme_assets/celeba/faces/00061.png)|![](readme_assets/celeba/yaw/73.gif)|![](readme_assets/celeba/pitch/55.gif)|![](readme_assets/celeba/lighting/25.gif)|
|![](readme_assets/celeba/faces/00062.png)|![](readme_assets/celeba/yaw/74.gif)|![](readme_assets/celeba/pitch/56.gif)|![](readme_assets/celeba/lighting/26.gif)|
|![](readme_assets/celeba/faces/00063.png)|![](readme_assets/celeba/yaw/75.gif)|![](readme_assets/celeba/pitch/57.gif)|![](readme_assets/celeba/lighting/27.gif)|
|![](readme_assets/celeba/faces/00064.png)|![](readme_assets/celeba/yaw/76.gif)|![](readme_assets/celeba/pitch/58.gif)|![](readme_assets/celeba/lighting/28.gif)|
|![](readme_assets/celeba/faces/00065.png)|![](readme_assets/celeba/yaw/77.gif)|![](readme_assets/celeba/pitch/59.gif)|![](readme_assets/celeba/lighting/29.gif)|
|![](readme_assets/celeba/faces/00066.png)|![](readme_assets/celeba/yaw/78.gif)|![](readme_assets/celeba/pitch/60.gif)|![](readme_assets/celeba/lighting/30.gif)|
|![](readme_assets/celeba/faces/00067.png)|![](readme_assets/celeba/yaw/79.gif)|![](readme_assets/celeba/pitch/61.gif)|![](readme_assets/celeba/lighting/31.gif)|
|![](readme_assets/celeba/faces/00068.png)|![](readme_assets/celeba/yaw/80.gif)|![](readme_assets/celeba/pitch/62.gif)|![](readme_assets/celeba/lighting/32.gif)|

### AFHQ Cat Experiments

#### Face Generation
| Original | Reproduced |
|----------|------------|
| ![](readme_assets/AFHQ_cat/faces/00010.png) | ![](readme_assets/AFHQ_cat/faces/00010_RE.png) |
| ![](readme_assets/AFHQ_cat/faces/00011.png) | ![](readme_assets/AFHQ_cat/faces/00011_RE.png) |
| ![](readme_assets/AFHQ_cat/faces/00012.png) | ![](readme_assets/AFHQ_cat/faces/00012_RE.png) |
| ![](readme_assets/AFHQ_cat/faces/00013.png) | ![](readme_assets/AFHQ_cat/faces/00013_RE.png) |
| ![](readme_assets/AFHQ_cat/faces/00014.png) | ![](readme_assets/AFHQ_cat/faces/00014_RE.png) |
| ![](readme_assets/AFHQ_cat/faces/00015.png) | ![](readme_assets/AFHQ_cat/faces/00015_RE.png) |

### Viewpoint Manipulation (yaw)
| Original | Reproduced |
|----------|------------|
| ![](readme_assets/AFHQ_cat/yaw/28.gif) | ![](readme_assets/AFHQ_cat/yaw/28_RE.gif)|
| ![](readme_assets/AFHQ_cat/yaw/29.gif) | ![](readme_assets/AFHQ_cat/yaw/29_RE.gif)|
| ![](readme_assets/AFHQ_cat/yaw/30.gif) | ![](readme_assets/AFHQ_cat/yaw/30_RE.gif)|
| ![](readme_assets/AFHQ_cat/yaw/31.gif) | ![](readme_assets/AFHQ_cat/yaw/31_RE.gif)|
| ![](readme_assets/AFHQ_cat/yaw/32.gif) | ![](readme_assets/AFHQ_cat/yaw/32_RE.gif)|
| ![](readme_assets/AFHQ_cat/yaw/33.gif) | ![](readme_assets/AFHQ_cat/yaw/33_RE.gif)|

### Viewpoint Manipulation (pitch)
| Original | Reproduced |
|----------|------------|
| ![](readme_assets/AFHQ_cat/pitch/64.gif) | ![](readme_assets/AFHQ_cat/pitch/64_RE.gif) |
| ![](readme_assets/AFHQ_cat/pitch/65.gif) | ![](readme_assets/AFHQ_cat/pitch/65_RE.gif) |
| ![](readme_assets/AFHQ_cat/pitch/66.gif) | ![](readme_assets/AFHQ_cat/pitch/66_RE.gif) |
| ![](readme_assets/AFHQ_cat/pitch/67.gif) | ![](readme_assets/AFHQ_cat/pitch/67_RE.gif) |
| ![](readme_assets/AFHQ_cat/pitch/68.gif) | ![](readme_assets/AFHQ_cat/pitch/68_RE.gif) |
| ![](readme_assets/AFHQ_cat/pitch/69.gif) | ![](readme_assets/AFHQ_cat/pitch/69_RE.gif) |


### Re-lighting
| Original | Reproduced |
|----------|------------|
| ![](readme_assets/AFHQ_cat/light/73.gif) | ![](readme_assets/AFHQ_cat/light/73_RE.gif) |
| ![](readme_assets/AFHQ_cat/light/74.gif) | ![](readme_assets/AFHQ_cat/light/74_RE.gif) |
| ![](readme_assets/AFHQ_cat/light/75.gif) | ![](readme_assets/AFHQ_cat/light/75_RE.gif) |
| ![](readme_assets/AFHQ_cat/light/76.gif) | ![](readme_assets/AFHQ_cat/light/76_RE.gif) |
| ![](readme_assets/AFHQ_cat/light/77.gif) | ![](readme_assets/AFHQ_cat/light/77_RE.gif) |
| ![](readme_assets/AFHQ_cat/light/78.gif) | ![](readme_assets/AFHQ_cat/light/78_RE.gif) |

