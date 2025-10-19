# EMRRG:Efficient Fine-Tuning Pre-trained X-ray MambaNetworks for Radiology Report Generation 

## Introduction
### Abstract
X-ray image-based medical report generation (MRG) is a pivotal area in artificial intelligence that can significantly reduce diagnostic burdens for clinicians and patient wait times. Existing MRG models predominantly rely on Large
Language Models (LLMs) to improve report generation, with limited exploration of pre-trained vision foundation models or advanced fine-tuning techniques. Mainstream frameworks either avoid fine-tuning or utilize simplistic methods like
LoRA, often neglecting the potential of enhancing cross-attention mechanisms. Additionally, while Transformer-based models dominate vision-language tasks, non-Transformer architectures, such as the Mamba network, remain underexplored for medical report generation, presenting a promising avenue for future research. In this paper, we propose EMRRG, a novel X-ray report generation framework that fine-tunes pre-trained Mamba networks using parameter-efficient methods. Specifically, X-ray images are divided into patches, tokenized, and processed by an SSM-based vision backbone for feature extraction, with Partial LoRA yielding optimal performance. An LLM with a hybrid decoder generates the medical report, enabling end-to-end training and achieving strong results on benchmark datasets. Extensive experiments on three widely used benchmark datasets fully validated the effectiveness of our proposed strategies for
the X-ray MRG. 

### Overview
The overall framework of our model is as follows：
![overview](https://github.com/Event-AHU/Medical_Image_Analysis/blob/main/EMRRG/framework.jpg)
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

## Results and Checkpoints(For CheXpert_plus as example)
| name |  model |
|:---:|:---:|
| EMRRG-CXP | [model](https://drive.google.com/file/d/13JIZ4IgMI4OwblemX-5KyNucD30jBKk9/view?usp=sharing) |


## Acknowledgement

+ [MambaXray_VL](https://github.com/Event-AHU/Medical_Image_Analysis/tree/main/CXPMRG_Bench_MambaXray_VL) Our work is based on the MambaXray_VL framework.

 
