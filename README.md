# Progressive Temporal Feature Alignment Network for Video Inpainting

This work is accepted in **CVPR2021** as Poster. It proposed a new video inpainting approach that combines temporal convolution as well as optical flow approach. 

Noted: This code is currently a beta version. Not gurantee to be fully correct.

## Update
[Optical Flow Davis](https://drive.google.com/file/d/1oi7cEGM0vFy5cWAdglvXi_r2g7ewuMPG/view?usp=sharing)
[Optical Flow FVI](https://drive.google.com/file/d/1DPsxfXcp_JRnfMTK7sOd6o7-w-NEaMiA/view?usp=sharing)
[Mask Davis](https://drive.google.com/file/d/1wFGea38g0SktmWymk1If5h3ESbxRvA66/view?usp=sharing)
[Mask FVI](https://drive.google.com/file/d/1FHFwuYjY8ovqSvmnt_cKtHW8VO_TgR_D/view?usp=sharing)
[Checkpoint](https://drive.google.com/file/d/17PN5aB34M-TkGP5jwCrcQRo76TNbfMu-/view?usp=sharing)

## Installation
```
torch==1.7.0
torchvision==0.8.1
```

## Dataset
For FVI dataset, please refer to https://github.com/amjltc295/Free-Form-Video-Inpainting.
For DAVIS dataset, please refer to https://davischallenge.org/.

## File Structure
```
TSAM
└── data
    ├── checkpoints
    ├── model_weights
    ├── results
    ├── FVI
    ├── DAVIS    
    └── runs
└── code
    └── master
        └── TSAM
            └── ...
```

## Prepare pretrained weights for training

Pretrained weights: download all the pretrained weights and put it under TSAM/data/model_weights
| Model Name                       |            | 
|----------------------------------|------------|
| TSM_imagenet_resent50_gated.pth | [weight](https://drive.google.com/file/d/1qTasW27vfxV80eIxuK--F-yfKtp-bc9F/view?usp=sharing) |
| TSM_imagenet_resent50.pth | [weight](https://drive.google.com/file/d/1rj3ualxhVO_1McAWJL8p2mjib0GdgMFD/view?usp=sharing) |


## Training
**FVI TSM moving object/curve masks:**
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py --config config/config_pretrain.json --dataset_config dataset_configs/FVI_all_masks.json
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py --config config/config_finetune.json --dataset_config dataset_configs/FVI_all_masks.json
```

## Testing
Change the train.py in training scripts to test.py, and add ```-p /pth/to/ckpt``` to the end.

**DAVIS TSAM object removal:**
```
CUDA_VISIBLE_DEVICES=0 python3 test.py --config config/config_finetune_davis.json --dataset_config dataset_configs/DAVIS_removal.json -p /pth/to/ckpt
```

## Citation
```
@inproceedings{zou2020progressive,
  title={Progressive Temporal Feature Alignment Network for Video Inpainting},
  author={Xueyan Zou and Linjie Yang and Ding Liu and Yong Jae Lee},
  booktitle={CVPR},
  year={2021}
}
```

## Acknowledgement
Part of the code is borrow from https://github.com/amjltc295/Free-Form-Video-Inpainting and https://github.com/researchmm/STTN. Thanks for their great works!
