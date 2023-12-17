![Python >=3.6](https://img.shields.io/badge/Python->=3.6-yellow.svg)
![PyTorch >=1.7](https://img.shields.io/badge/PyTorch->=1.7-blue.svg)

# Catalyst for Clustering-based Unsupervised Object Re-Identification: Feature Calibration [[pdf]](wating)
The *official* repository for [Catalyst for Clustering-based Unsupervised Object Re-Identification: Feature Calibration](wating).

## Requirements

### Installation
```bash
pip install -r requirements.txt
```
We recommend to use /Python=3.8 /torch=1.10.1 /torchvision=0.11.2 /timm=0.6.13 /cuda==11.3 /faiss-gpu=1.7.2/ 24G RTX 3090 or RTX 4090 for training and evaluation. If you find some packages are missing, please install them manually. 

### Prepare Datasets

```bash
mkdir data
```

Download the datasets:
- [Market-1501](https://drive.google.com/file/d/1pYM3wruB8TonHLwMQ_g1KAz-UqRrH006/view?usp=drive_link)
- [MSMT17](https://drive.google.com/file/d/1TD3COX3laYIpXNvKN6vazv_7x8PNdYkI/view?usp=drive_link)
- [LUPerson](https://github.com/DengpanFu/LUPerson).  
- We don't have the copyright of the LUPerson dataset. Please contact authors of LUPerson to get this dataset.
- You can download the file list ordered by the CFS score for the LUPerson. [[CFS_list.pkl]](https://drive.google.com/file/d/1D6RaiOv3F2WSABYfQB1Aa88mwGoVNa3k/view?usp=sharing)

Then unzip them and rename them under the directory like

```
data
├── market1501
│   └── bounding_box_train
│   └── bounding_box_test
│   └── ..
├── MSMT17
│   └── train
│   └── test
│   └── ..
└── DukeMTMC-reID
    └── bounding_box_train
    └── bounding_box_test
    └── query
    └── *.txt
```

## Pre-trained Models 
| Model         | Download |
| :------:      | :------: |
| ViT-S/16      | [link](https://drive.google.com/file/d/1ODxA7mJv17UfzwfXtY9dTWNsYghoNWGB/view?usp=sharing) |
| ViT-S/16+ICS  | [link](https://drive.google.com/file/d/18FL9JaJNlo15-UksalcJRXX-0dgo4Mz4/view?usp=sharing) |
| ViT-B/16+ICS  | [link](https://drive.google.com/file/d/1ZFMCBZ-lNFMeBD5K8PtJYJfYEk5D9isd/view?usp=sharing) |

## Pre-trained Models on Baseline
| Model         | Download |
| :------:      | :------: |
| Market-1501      | [link](https://drive.google.com/file/d/1B3qKOdlfpJ3BujxNY0bhhduFl9UgaW8H/view?usp=drive_link) |
| MSMT17  | [link](https://drive.google.com/file/d/1aJLYqP4XP6uDgL4LxVGf6ynuq5252T8p/view?usp=drive_link) |
| DukeMTMC-reID  | [link](https://drive.google.com/file/d/1AVsYiATHMVK_0761kgWMzvHaX4d90iwP/view?usp=drive_link) |


Please download pre-trained models and put them into your custom file path.

## Examples
### ViT
`sh train.sh`


## ReID performance

We have reproduced the performance to verify the reproducibility. The reproduced results may have a gap of about 0.5% with the numbers in the paper.


### USL ReID
  
##### Market-1501
| Model         | Image Size| mAP/Rank-1 | Download |
| :------:      | :------: |:------: |:------: |
| ViT-S/16      | 256*128 |88.6/94.9 |[model](https://drive.google.com/file/d/1uILUIumfH19PZ1hWcvMIvv46k0znM3aH/view?usp=drive_link)| 



##### MSMT17
| Model         | Image Size| mAP/Rank-1 | Download |
| :------:      | :------: |:----------:|:------: |
| ViT-S/16      | 256*128 | 49.7/74.8  |[model](https://drive.google.com/file/d/1FCbQ4aXBc6aS1bAADYWA6uF0Zg8m2my6/view?usp=drive_link)| 


##### DukeMTMC-reID
| Model         | Image Size| mAP/Rank-1 | Download |
| :------:      | :------: |:----------:|:------: |
| ViT-S/16      | 256*128 | 71.9/83.8  |[model](https://drive.google.com/file/d/1NzEz0FmeR_CDLNyGTmD8r4Qw1jqCZLt5/view?usp=drive_link)| 





## Acknowledgment
Our implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.

[LUPerson](https://github.com/DengpanFu/LUPerson), [DINO](https://github.com/facebookresearch/dino), [TransReID](https://github.com/damo-cv/TransReID), [cluster-contrast-reid](https://github.com/alibaba/cluster-contrast-reid),
[TransReID-SSL](https://github.com/damo-cv/TransReID-SSL)

## Citation

If you find this code useful for your research, please cite our paper

```
wating
```

## Contact

If you have any question, please feel free to contact us. E-mail: [qingsonghu08@gmail.com](mailto:qingsonghu08@gmail.com)
