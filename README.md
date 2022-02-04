# [Re] - Lifting 2D StyleGAN for 3D-Aware Face Generation

This repository is the re-production implementation of [Lifting 2D StyleGAN for 3D-Aware Face Generation](https://arxiv.org/abs/2011.13126) by [Yichun Shi](https://seasonsh.github.io), [Divyansh Aggarwal](https://divyanshaggarwal.github.io)and Anil K. Jain in the scope of [ML Reproducibility Challenge 2021](https://paperswithcode.com/rc2021).
<!---
Authored by [Doğa Yılmaz](https://yilmazdoga.com), [Furkan Kınlı](https://birdortyedi.github.io/), Barış Özcan, [Furkan Kıraç](http://fkirac.net/).
--->

## Requirements

You can create the conda environment by using:
```setup
conda env create -f environment.yml
```

## Training
### Training from pre-trained StyleGAN2
Download our pre-trained StyleGAN and face embedding network from [here](https://drive.google.com/file/d/1qVoWu_fps17iTzYptwuN3ptgYeCpIl2e/view?usp=sharing) for training. Unzip them into the `pretrained/` folder. Then you can start training by:
```sh
python tools/train.py config/ffhq_256.py
```

### Training from custom data
We use a re-cropped version of FFHQ to fit the style of our face embedding network. You can find this dataset [here](https://drive.google.com/file/d/1pLHzbZS52XGyejubv5tT0CqhpsocaYuD/view?usp=sharing). The cats dataset can be found [here](https://drive.google.com/file/d/1soEXvvLV0uhasg9GlVhH5YW_9FsAmb3d/view?usp=sharing).
To train a StyleGAN2 from you own dataset, check the content under [`stylegan2-pytorch`](https://github.com/seasonSH/LiftedGAN/tree/main/stylegan2-pytorch) folder. After training a StyleGAN2, you can lift it using our training code.

## Testing
### Pre-trained Models: 
[Google Drive](https://drive.google.com/file/d/1-44Eivt7GHINkX6zox89HHttujYWThz2/view?usp=sharing)
### Sampling random faces
You can generate random samples from a lifted gan by running:
```sh
python tools/generate_images.py /path/to/the/checkpoint --output_dir results/
```
Make sure the checkpoint file and its `config.py` file are under the same folder.

### Testing FID
We use the code from rosinality's stylegan2-pytorch to compute FID. To compute the FID, you first need to compute the statistics of real images:
```sh
python utils/calc_inception.py /path/to/the/dataset/lmdb
```
You might skip this step if you are using our pre-calculated statistics file ([link](https://drive.google.com/file/d/1qVoWu_fps17iTzYptwuN3ptgYeCpIl2e/view?usp=sharing)). Then, to test the FID, you can run:
```sh
python tools/test_fid.py /path/to/the/checkpoint --inception /path/to/the/inception/file
```

## Additional Results to Our Reproduction Paper

### FFHQ Experiments

#### Faces Yaw
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


### Faces Yaw
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


### Faces Pitch
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

### Faces Lighting
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

| Face Generation | Viewpoint Manipualtion (yaw) | Viewpoint Manipualtion (pitch) | Re-lighting |
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

