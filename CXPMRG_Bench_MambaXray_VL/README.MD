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
