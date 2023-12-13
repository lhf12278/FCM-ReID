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

We trained the CC-Res on two RTX 3090.

### Prepare Datasets

```bash
mkdir data
```

Download the datasets: 修改1：链接
- [Market-1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view)
- [MSMT17](https://arxiv.org/abs/1711.08565)
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
└── VeRi
    └── image_train
    └── image_query
    └── image_test
    └── *.txt
```

## Pre-trained Models 
| Model         | Download |
| :------:      | :------: |
| ViT-S/16      | [link](https://drive.google.com/file/d/1ODxA7mJv17UfzwfXtY9dTWNsYghoNWGB/view?usp=sharing) |
| ViT-S/16+ICS  | [link](https://drive.google.com/file/d/18FL9JaJNlo15-UksalcJRXX-0dgo4Mz4/view?usp=sharing) |
| ViT-B/16+ICS  | [link](https://drive.google.com/file/d/1ZFMCBZ-lNFMeBD5K8PtJYJfYEk5D9isd/view?usp=sharing) |

Please download pre-trained models and put them into your custom file path.

## Examples

### ResNet-50
`cd cluster-contrast-reid-res`  
`sh train.sh`


### ViT
`cd cluster-contrast-reid-vit`
`sh train.sh`


## ReID performance

We have reproduced the performance to verify the reproducibility. The reproduced results may have a gap of about 1.0% with the numbers in the paper.


### USL ReID  修改3 ：wating：
  
##### Market-1501
| Model         | Image Size| mAP/Rank-1 | Download |
| :------:      | :------: |:------: |:------: |
| ViT-S/16      | 256*128 |89.6/94.5 |  |[model](https://drive.google.com/| 
| ResNet-50      | 256*128 |86.9/94.5 |[model](https://drive.google.com/)| 


##### MSMT17
| Model         | Image Size| mAP/Rank-1 | Download |
| :------:      | :------: |:------: |:------: |
| ViT-S/16      | 256*128 |58.5/79.7 |[model](https://drive.google.com/)| 
| ResNet-50     | 256*128 |55.9/78.8 |[model](https://drive.google.com/)| 

##### VeRi-776
| Model         | Image Size| mAP/Rank-1 | Download |
| :------:      | :------: |:------: |:------: |
| ResNet-50     | 224*224 |47.8/91.4 |[model](https://drive.google.com/)| 



## Acknowledgment
Our implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.

[LUPerson](https://github.com/DengpanFu/LUPerson), [DINO](https://github.com/facebookresearch/dino), [TransReID](https://github.com/damo-cv/TransReID), [cluster-contrast-reid](https://github.com/alibaba/cluster-contrast-reid),
[TransReID-SSL](https://github.com/damo-cv/TransReID-SSL)

## Citation

If you find this code useful for your research, please cite our paper  修改4：引用链接

```
wating
```

## Contact  修改5

If you have any question, please feel free to contact us. E-mail: [qingsonghu08@gmail.com](mailto:qingsonghu08@gmail.com)
