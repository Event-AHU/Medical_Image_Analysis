# R2GenCSR: Retrieving Context Samples for Large Language Model based X-ray Medical Report Generation

## Introduction
![overview](https://github.com/Event-AHU/Medical_Image_Analysis/blob/main/R2GenCSR/images/framework.jpg)

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



```bash
cheXpert_plus 
├── chexbert_labels
├── df_chexpert_plus_240401.csv
├── PNGs
├── annotation.json    <- place annotation.json here
```



### Prepare Vmamba:
Follow the instruction in [VMamba](https://github.com/MzeroMiko/VMamba) to prepare the Vmamba model, you can download Vmamba-base from [here](https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm_base_0229_ckpt_epoch_237.pth). 

Install selective_scan kernel for your environment:
```bash
cd VMamba/kernels/selective_scan && pip install .
```

### Training

For MIMIC-CXR as example, update '--annotation' and '--base_dir' in scripts/mimic.sh to your data path.

For IU-xray:
```bash
bash scripts/iu.sh
```

For MIMIC-CXR:
```bash
bash scripts/mimic.sh
```

For CheXpert Plus:
```bash
bash scripts/chexplus.sh
```


### Testing (For MIMIC-CXR as example)

**Testing mode**

Once you finished the training, you can test the model by running the following method:

```bash
bash scripts/mimic.sh path/to/save/checkpoint.pth
```

You can download the models we trained for each dataset from [here](https://drive.google.com/drive/folders/18mx2XMrWGc6MqKxFuQQUWRVp4E_YIaQu?usp=drive_link) to test the performance. Note that different devices and batchsize may lead to varying results.

**Domo mode**

For MIMIC-CXR, update '--delta_file' in scripts/mimic_demo.sh to checkpoint path, and run:

```bash
bash scripts/mimic_demo.sh
```
Demo video:
https://github.com/Event-AHU/Medical_Image_Analysis/blob/main/R2GenCSR/images/demo.mp4

## Acknowledgement

+ [Vmamba](https://github.com/MzeroMiko/VMamba) A vision backbone that works in linear time complexity. We have changed the original code to support BF16 Vmamba.

+ [R2GenGPT](https://github.com/wang-zhanyu/R2GenGPT/tree/main) Our work is based on the R2GenGPT framework.


## License
This repository is under [BSD 3-Clause License](LICENSE.md).
