# [CVPR2025-Expand] Pre-training, Reasoning, Benchmarking: X-ray Report Generation on CheXpert Plus Dataset 

## Introduction
### Abstract

X-ray image-based Radiology Report Generation (RRG) constitutes a critical research direction within medical artificial intelligence, with great potential to alleviate clinicians’ diagnostic workload and shorten patient waiting periods. Despite substantial advances over recent years, the field faces evident bottlenecks stemming from insufficient standardized benchmarks and inadequate domain adaptation of generic large models. Notably, the newly released CheXpert Plus dataset is provided without accompanying baseline implementations and evaluation results, which impedes standardized training, quantitative evaluation and fair comparison among follow-up algorithms. To mitigate this limitation, we establish a comprehensive benchmark encompassing prevailing X-ray report generation models and Large Language Models on CheXpert Plus. This benchmark delivers a reliable comparative foundation for upcoming methods and enables researchers to rapidly identify state-of-the-art approaches within this domain. Beyond benchmark construction, we rethink X-ray RRG under the paradigm of large models and propose a novel framework termed MambaXray-PRB. Our framework improves report generation performance and enhances model interpretability via multi-stage large-model pre-training and multi-modal Chain-of-Thought reasoning. The pipeline consists of three successive phases: self-supervised auto-regressive modeling, X-ray–report contrastive learning, and post-training optimization for reasoning and report generation. Extensive experiments on IU X-ray, MIMIC-CXR, and CheXpert Plus datasets validate the effectiveness of MambaXray-PRB for radiology report generation. 

![overall](https://github.com/Event-AHU/Medical_Image_Analysis/blob/main/MambaXray_PRB/benchmarkResults_Vis.png)

### Overview
The overall framework of our model is as follows：
![CoT-data](https://github.com/Event-AHU/Medical_Image_Analysis/blob/main/MambaXray_PRB/CoT_data.jpg)

![overview](https://github.com/Event-AHU/Medical_Image_Analysis/blob/main/MambaXray_PRB/framework.jpg)

## Getting Started
### Installation

**1. Install requirements**

Install requirements using pip:

```bash
pip install -r requirements.txt
```


**2. Prepare dataset**

We follow R2Gen dataset process to download [IU-Xray](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view) and [MIMIC-CXR](https://drive.google.com/file/d/1DS6NYirOXQf8qYieSVMvqNwuOlgAbM_E/view?usp=sharing), you can download orin MIMIC-CXR dataset from [official website](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)

For CheXpert Plus dataset: Download the original CheXpert Plus dataset from [here](https://stanfordaimi.azurewebsites.net/datasets/5158c524-d3ab-4e02-96e9-6ee9efc110a1).

The data set we used in the first stage of training was an undisclosed private data set, so I can't provide it here. However, you can use other medical X-ray data for training. In the second stage we use a collection of three public data sets [IU-Xray](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view), [Mimic-CXR](https://physionet.org/content/mimic-cxr-jpg/2.0.0/), and [CheXpert-plus](https://stanfordaimi.azurewebsites.net/datasets/5158c524-d3ab-4e02-96e9-6ee9efc110a1). You can download the three raw data and then use the annotation files we provide in [this link](https://drive.google.com/drive/folders/14ASXUwGU9SQqYW1WrK8eJdpqwzNrYhTX?usp=drive_link) and [this link](https://drive.google.com/drive/folders/17W0w4FNY0tw5GfUTE7IhGpGSHiC3tFfp?usp=drive_link). The third stage of downstream task fine-tuning uses separate, normal three public data sets. Place the annotation file in the project directory and specify its path in the `.sh` file.

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
| IU-Finetune-B       | 3 | Base  | [model](https://drive.google.com/file/d/1TGjk0c_vcuEIEhIkNo0wgj5WHHKle8Ck/view?usp=drive_link) |
| IU-Finetune-L       | 3 | Large | [model](https://drive.google.com/file/d/1ZjOovSQIq7Nz9X3y3Rd9XwWNZWPjAJTd/view?usp=drive_link) |
| Mimic-Finetune-B    | 3 | Base  | [model](https://drive.google.com/file/d/1qxyn2_8itqu3NjNfOtzuETJHuPpCxLH1/view?usp=drive_link) |
| Mimic-Finetune-L    | 3 | Large | [model](https://drive.google.com/file/d/1aU289WvsQ4JnuOHXqsMfKTEcg5T2H0-D/view?usp=drive_link) |
| CheXpert-Finetune-B | 3 | Base  | [model](https://drive.google.com/file/d/1ezsLhW4_y_w5lJAkunoDpPG-glZ8MUOP/view?usp=drive_link) |
| CheXpert-Finetune-L | 3 | Large | [model](https://drive.google.com/file/d/1a-ZCM_JI0PJqqi5RrwfU0138sqwOctvv/view?usp=drive_link) |

## Acknowledgement

+ [ARM](https://github.com/OliverRensu/ARM) The first stage of training of our model is based on ARM.

+ [R2GenGPT](https://github.com/wang-zhanyu/R2GenGPT/tree/main) Our work is based on the R2GenGPT framework.


## Citation
If you find this repository helpful, please consider citing:
```bibtex
come soon……
```

