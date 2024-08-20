<p align="center">
<img src="https://github.com/Event-AHU/Medical_Image_Analysis/blob/main/logo.jpg" width="400">
</p>




## Update Log:  

* [2024.08.20] **R2GenCSR: Retrieving Context Samples for Large Language Model based X-ray Medical Report Generation** is released [arXiv:2408.09743]

* [2024.04.28] **Pre-training on High Definition X-ray Images: An Experimental Study**, Xiao Wang, Yuehang Li, Wentao Wu, Jiandong Jin, Yao Rong, Bo Jiang, Chuanfu Li, Jin Tang
arXiv pre-print, arXiv 2024, is released on **arXiv** [[Paper](https://arxiv.org/abs/2404.17926)]


## Surveys/Reviews 
* **Automated Radiology Report Generation: A Review of Recent Advances**, Phillip Sloan, Philip Clatworthy, Edwin Simpson, Majid Mirmehdi
  [[Paper](https://arxiv.org/abs/2405.10842)]



## :dart: [Pre-training MAE Model on HD X-ray Images]() 
* **R2GenCSR: Retrieving Context Samples for Large Language Model based X-ray Medical Report Generation**, 
Xiao Wang, Yuehang Li, Fuling Wang, Shiao Wang, Chuanfu Li, Bo Jiang, 
arXiv Pre-print arXiv:2408.09743, 2024 
[[Paper](https://arxiv.org/abs/2408.09743)] 
Inspired by the tremendous success of Large Language Models (LLMs), existing X-ray medical report generation methods attempt to leverage large models to achieve better performance. They usually adopt a Transformer to extract the visual features of a given X-ray image, and then, feed them into the LLM for text generation. How to extract more effective information for the LLMs to help them improve final results is an urgent problem that needs to be solved. Additionally, the use of visual Transformer models also brings high computational complexity. To address these issues, this paper proposes a novel context-guided efficient X-ray medical report generation framework. Specifically, we introduce the Mamba as the vision backbone with linear complexity, and the performance obtained is comparable to that of the strong Transformer model. More importantly, we perform context retrieval from the training set for samples within each mini-batch during the training phase, utilizing both positively and negatively related samples to enhance feature representation and discriminative learning. Subsequently, we feed the vision tokens, context information, and prompt statements to invoke the LLM for generating high-quality medical reports. Extensive experiments on three X-ray report generation datasets (i.e., IU-Xray, MIMIC-CXR, CheXpert Plus) fully validated the effectiveness of our proposed model. 

![R2GenCSR](https://github.com/Event-AHU/Medical_Image_Analysis/blob/main/R2GenCSR/framework.jpg)



## :dart: [Pre-training MAE Model on HD X-ray Images]() 
* **Pre-training on High Definition X-ray Images: An Experimental Study**, 
Xiao Wang, Yuehang Li, Wentao Wu, Jiandong Jin, Yao Rong, Bo Jiang, Chuanfu Li, Jin Tang
arXiv pre-print, arXiv, 2024 
[[Paper](https://arxiv.org/abs/2404.17926)] 
Existing X-ray based pre-trained vision models are usually conducted on a relatively small-scale dataset (less than 500k samples) with limited resolution (e.g., 224 × 224). However, the key to the success of self-supervised pre-training large models lies in massive training data, and maintaining high resolution in the field of X-ray images is the guarantee of effective solutions to difficult miscellaneous diseases. In this paper, we address these issues by proposing the first high-definition (1280 × 1280) X-ray based pre-trained foundation vision model on our newly collected large-scale dataset which contains more than 1 million X-ray images. Our model follows the masked auto-encoder framework which takes the tokens after mask processing (with a high rate) is used as input, and the masked image patches are reconstructed by the Transformer encoder-decoder network. More importantly, we introduce a novel context-aware masking strategy that utilizes the chest contour as a boundary for adaptive masking operations. We validate the effectiveness of our model on two downstream tasks, including X-ray report generation and disease recognition. Extensive experiments demonstrate that our pre-trained medical foundation vision model achieves comparable or even new state-of-the-art performance on downstream benchmark datasets.

![HDXrayPretrain](https://github.com/Event-AHU/Medical_Image_Analysis/blob/main/HD_Xray_Pretrain_MAE/framework.jpg)







## Paper Lists: 
* [pretraining_paper_list.md](https://github.com/Event-AHU/Medical_Image_Pretraining/blob/main/pretrain_paper_list.md)
* [Report_Error_checking.md](https://github.com/Event-AHU/Medical_Image_Pretraining/blob/main/Report_Error_checking.md)
* [medical_report_generation.md](https://github.com/Event-AHU/Medical_Image_Pretraining/blob/main/medical_report_generation.md) 
* [xray-classification.md](https://github.com/Event-AHU/Medical_Image_Pretraining/blob/main/xray-classification.md)
* [medical image-text understanding tasks](https://github.com/Event-AHU/Medical_Image_Pretraining/blob/main/medical_image-text_understanding_tasks.md)
* [medical_image_segmentation](https://github.com/Event-AHU/Medical_Image_Pretraining/blob/main/medical_image_segmentation.md)


## Suggested Code: 
* Wang, Zhanyu, et al. "**R2gengpt: Radiology report generation with frozen llms**." Meta-Radiology 1.3 (2023): 100033.
  [[Github](https://github.com/wang-zhanyu/R2GenGPT)]
  [[Paper](https://www.sciencedirect.com/science/article/pii/S2950162823000334)]
  
* **R2Gen**: [[Paper (EMNLP-2020)](https://arxiv.org/pdf/2010.16056.pdf)] [[Code](https://github.com/zhjohnchan/R2Gen)]
  



## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=Event-AHU/Medical_Image_Analysis&type=Date)](https://star-history.com/#Event-AHU/Medical_Image_Analysis&Date)







  
