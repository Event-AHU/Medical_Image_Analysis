# R2GenKG: Hierarchical Multi-modal Knowledge Graph for LLM-based Radiology Report Generation

## Introduction
![overview](https://github.com/Event-AHU/Medical_Image_Analysis/blob/main/R2GenKG/figures/framework_FulingWang.jpg)

## Getting Started
### Installation

**1. Install requirements**

Install requirements using pip:

```bash
pip install -r requirements.txt
```


**2. Prepare dataset**

We follow R2Gen dataset process to download [IU-Xray](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view) 
For CheXpert Plus dataset: Create a 'chexpert_plus' folder and download the original CheXpert Plus dataset from [here](https://stanfordaimi.azurewebsites.net/datasets/5158c524-d3ab-4e02-96e9-6ee9efc110a1). You can download our preprocess annotation file from [here](https://drive.google.com/file/d/1vjh8GXaFQYJXJeLaxLnFtvZxuSZscQd_/view?usp=sharing).



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
bash scripts/1-1.run_iuxray.sh
```

For CheXpert Plus:
```bash
bash scripts/6-1.cheXpert_plus_run.sh
```

### Testing 

**Testing mode**

Once you finished the training, you can test the model by running the following method:

For IU-xray:
```bash
bash scripts/1-2.test_iuxray.sh
```

For CheXpert Plus:
```bash
bash scripts/6-2.cheXpert_plus_test.sh


Note: Complete execution steps, code for processing graph features, and weights will be uploaded later.

## Acknowledgement

R2GenGPT Our work is based on the R2GenGPT framework.

