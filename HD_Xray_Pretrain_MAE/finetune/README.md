# HD_Xray_Pretrain_MAE
## Requirements
### Installation
To install the required dependencies, run:
```
pip install -r requirements.txt
```
### Data Preparation
For the Disease Prediction task, download the RSNA dataset. Check './DP/dataset/medical_classifi_pickle_process.py' to preprocess the RSNA dataset into a pickle format for training.

For the Report Generation task, download the IU-Xray and MIMIC-CXR datasets. Modify the '--ann_path' value in the corresponding '.sh' file located in the RG_chinese and RG_english directories. Download pycocoevalcap from [here]() and place it in the pycocoevalcap folder to calculate metrics. 

### Pre-trained Model
Download the High-Definition Pre-trained Vision Foundation Model from [here]() to use it for training.
## Training

To train the model for a specific task, run the corresponding '.sh' file.

For example, for the Disease Prediction task, navigate to the appropriate directory and execute the following commands:
```
cd ./finetune_download/DP
bash run_rsna-pneumonia100p.sh
```
This will start the training process using the specified configuration.
## Citation and Acknowledgements

This code is based on [R2Gen](https://github.com/zhjohnchan/R2Gen) and [VTB](https://github.com/cxh0519/VTB).