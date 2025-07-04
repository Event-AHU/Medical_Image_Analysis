# [CVPR2025] CXPMRG-Bench: Pre-training and Benchmarking for X-ray Medical Report Generation on CheXpert Plus Dataset 

## Introduction
### Abstract

X-ray image-based medical report generation (MRG) is a pivotal area in artificial intelligence which can significantly reduce diagnostic burdens and patient wait times. Despite significant progress, we believe that the task has reached a bottleneck due to the limited benchmark datasets and the existing large models' insufficient capability enhancements in this specialized domain. Specifically, the recently released CheXpert Plus dataset lacks comparative evaluation algorithms and their results, providing only the dataset itself. This situation makes the training, evaluation, and comparison of subsequent algorithms challenging. Thus, we conduct a comprehensive benchmarking of existing mainstream X-ray report generation models and large language models (LLMs), on the CheXpert Plus dataset. We believe that the proposed benchmark can provide a solid comparative basis for subsequent algorithms and serve as a guide for researchers to quickly grasp the state-of-the-art models in this field. More importantly, we propose a large model for the X-ray image report generation using a multi-stage pre-training strategy, including self-supervised autoregressive generation and Xray-report contrastive learning, and supervised fine-tuning. Extensive experimental results indicate that the autoregressive pre-training based on Mamba effectively encodes X-ray images, and the image-text contrastive pre-training further aligns the feature spaces, achieving better experimental results. 

![overall](https://github.com/Event-AHU/Medical_Image_Analysis/blob/main/CXPMRG_Bench_MambaXray_VL/CXPMRG_Bench.png)

### Overview
The overall framework of our model is as follows：
![overview](https://github.com/Event-AHU/Medical_Image_Analysis/blob/main/CXPMRG_Bench_MambaXray_VL/CXPMRG_benchmark02.png)

## Getting Started
### Installation

**1. Install requirements**

Install requirements using pip:

```bash
pip install -r requirements.txt
```


**2. Prepare dataset**

We follow R2Gen dataset process to download [IU-Xray](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view) and [MIMIC-CXR](https://drive.google.com/file/d/1DS6NYirOXQf8qYieSVMvqNwuOlgAbM_E/view?usp=sharing), you can download orin MIMIC-CXR dataset from [official website](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)

For CheXpert Plus dataset: Download the original CheXpert Plus dataset from [here](https://stanfordaimi.azurewebsites.net/datasets/5158c524-d3ab-4e02-96e9-6ee9efc110a1). You can download our preprocess annotation file from [here](https://drive.google.com/file/d/1vjh8GXaFQYJXJeLaxLnFtvZxuSZscQd_/view?usp=sharing).

The data set we used in the first stage of training was an undisclosed private data set, so I can't provide it here. However, you can use other medical X-ray data for training. In the second stage we use a collection of three public data sets [IU-Xray](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view), [Mimic-CXR](https://physionet.org/content/mimic-cxr-jpg/2.0.0/), and [CheXpert-plus](https://stanfordaimi.azurewebsites.net/datasets/5158c524-d3ab-4e02-96e9-6ee9efc110a1). You can download the three raw data and then use the annotation file we [provide](https://drive.google.com/file/d/1o3uGa__FRJQo0iC-By63XzIarsErgucb/view?usp=drive_link). The third stage of downstream task fine-tuning uses separate, normal three public data sets.

Using cheXpert_plus as an example, the file format of the data set is as follows:
```bash
cheXpert_plus 
├── chexbert_labels
├── df_chexpert_plus_240401.csv
├── PNGs
├── annotation.json    <- place annotation.json here
```

### Training

**1. Pretrain Stage**

You need to change '--data_path' in pretrain/pretrain.sh to your own data set configuration file. Then run the command:
```bash
bash pretrain/pretrain.sh
```

**2. MambaXrayVLCLIP Stage**

At this stage, you first need to download the [annotation](https://drive.google.com/file/d/1o3uGa__FRJQo0iC-By63XzIarsErgucb/view?usp=drive_link) file. Note that the information in this file is used in absolute path, so you need to modify the path corresponding to the three datasets you downloaded before using it. And you need to change '--annotation' in launch/launch_mambaclip.sh. Then run the command:
```bash
bash launch/launch_mambaclip.sh
```

**3. DownStream Stage**

For CheXpert_plus as example, update '--annotation' and '--base_dir' in launch/launch_mambaclip_chexpert.sh to your data path.

For IU-xray:
```bash
bash launch/launch_mambaclip_iu.sh
```

For MIMIC-CXR:
```bash
bash launch/launch_mambaclip_mimic.sh
```

For CheXpert Plus:
```bash
bash launch/launch_mambaclip_chexpert.sh
```


### Testing (For CheXpert_plus as example)

Once you finished the training, update '--annotation' , '--base_dir' and 'load_model' in launch/launch_mambaclip_test_chexpert.sh to your data path, then you can test the model by running the following method:

```bash
bash launch/launch_mambaclip_test_chexpert.sh
```

## Results and Checkpoints
| name | stage | #params | model |
|:---:|:---:|:---:|:---:|
| Pretrain-B          | 1 | Base  | [model](https://drive.google.com/file/d/17hQy_VAChRIXySUslUexYY8HlNGXXFrY/view?usp=sharing) |
| Pretrain-L          | 1 | Large | [model](https://drive.google.com/file/d/1TTpG5LIzngbuWxjTXWdhhcRNPEHlMJkG/view?usp=sharing) |
| MambaXrayCLIP-B     | 2 | Base  | [model](https://drive.google.com/file/d/16BqLKpvfLghLB7-ijoVbanFmsu2Y9tit/view?usp=sharing) |
| MambaXrayCLIP-L     | 2 | Large | [model](https://drive.google.com/file/d/1borqq55uoBYTR6lRfowdG4u3mSOWUnJ7/view?usp=sharing) |
| IU-Finetune-B       | 3 | Base  | [model](https://drive.google.com/file/d/1Dw4YOUZjMSyZ-N84MfhdHoJIw6B_-_lC/view?usp=sharing) |
| IU-Finetune-L       | 3 | Large | [model](https://drive.google.com/file/d/1eVR5LAkvWaj3wCOHnvoXg6ArbHZy07sl/view?usp=sharing) |
| Mimic-Finetune-B    | 3 | Base  | [model](https://drive.google.com/file/d/1toF888tdpxPKX7h1A-QFCL7Itega7L-k/view?usp=sharing) |
| Mimic-Finetune-L    | 3 | Large | [model](https://drive.google.com/file/d/124d-FeJyuShFA7x6_2SBrMMx20Pj4OKX/view?usp=sharing) |
| CheXpert-Finetune-B | 3 | Base  | [model](https://drive.google.com/file/d/13JIZ4IgMI4OwblemX-5KyNucD30jBKk9/view?usp=sharing) |
| CheXpert-Finetune-L | 3 | Large | [model](https://drive.google.com/file/d/1FfljNgp4PKSeiqsT2CeSCSA1WlbjrNng/view?usp=sharing) |

## Acknowledgement

+ [ARM](https://github.com/OliverRensu/ARM) The first stage of training of our model is based on ARM.

+ [R2GenGPT](https://github.com/wang-zhanyu/R2GenGPT/tree/main) Our work is based on the R2GenGPT framework.


## Citation
If you find this repository helpful, please consider citing:
```bibtex
@InProceedings{Wang_2025_CVPR,
    author    = {Wang, Xiao and Wang, Fuling and Li, Yuehang and Ma, Qingchuan and Wang, Shiao and Jiang, Bo and Tang, Jin},
    title     = {CXPMRG-Bench: Pre-training and Benchmarking for X-ray Medical Report Generation on CheXpert Plus Dataset},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {5123-5133}
}
```
