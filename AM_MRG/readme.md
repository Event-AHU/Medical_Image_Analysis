# Activating Associative Disease-Aware Vision Token Memory for LLM-Based X-ray Report Generation

## Introduction
![overview](https://github.com/Event-AHU/Medical_Image_Analysis/blob/main/AM_MRG/figures/framework_FulingWang.jpg)

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



### Prepare MambaXray-VL:
Follow the instruction in [MambaXray-VL](https://github.com/Event-AHU/Medical_Image_Analysis/tree/main/CXPMRG_Bench_MambaXray_VL) to prepare the MambaXray-VL model. 

### Download the necessary Checkpoint files:
You can download the necessary weight files for AM_MRG [here](https://www.dropbox.com/scl/fo/gtgo8hs2qxwdldbxlownu/ADIthMRX6w34BBCUpVYByRU?rlkey=ahdpc4knewqyvdxjv65ghdmbj&st=x4ubtmy6&dl=0) (CAM.pkl, Report_Memory.pkl, and Stage1ckpt.pth).
The weight files required for QFormer can be found in the [BLIP2](https://dl.acm.org/doi/10.5555/3618408.3619222) paper.


### Training

For MIMIC-CXR as example, update '--annotation' and '--base_dir' in scripts/mimic.sh to your data path.

For IU-xray:
```bash
bash scripts/train_iu_xray.sh
```

For MIMIC-CXR:
```bash
bash scripts/train_mimic_cxr.sh
```

For CheXpert Plus:
```bash
bash scripts/train_chexpert_plus.sh
```


### Testing (For MIMIC-CXR as example)

**Testing mode**

Once you finished the training, you can test the model by running the following method:

```bash
bash scripts/test_mimic_cxr.sh
```



## Acknowledgement

+ [MambaXray-VL](https://github.com/Event-AHU/Medical_Image_Analysis/tree/main/CXPMRG_Bench_MambaXray_VL) A vision backbone that works in linear time complexity. 

+ [R2GenGPT](https://github.com/wang-zhanyu/R2GenGPT/tree/main) Our work is based on the R2GenGPT framework.


## License
This repository is under [BSD 3-Clause License](LICENSE.md).



