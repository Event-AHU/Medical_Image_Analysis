# CXPMRG-Bench: Pre-training and Benchmarking for X-ray Medical Report Generation on CheXpert Plus Dataset

## Introduction
![overall](https://github.com/Event-AHU/Medical_Image_Analysis/blob/main/CXPMRG_Bench_MambaXray_VL/CXPMRG_Bench.png)
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

For CheXpert Plus dataset: Create a 'chexpert_plus' folder and download the original CheXpert Plus dataset from [here](https://stanfordaimi.azurewebsites.net/datasets/5158c524-d3ab-4e02-96e9-6ee9efc110a1). You can download our preprocess annotation file from [here](https://drive.google.com/file/d/1vjh8GXaFQYJXJeLaxLnFtvZxuSZscQd_/view?usp=sharing).
