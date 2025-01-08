<p align="center">
<img src="https://github.com/Event-AHU/Medical_Image_Analysis/blob/main/logo.jpg" width="400">
</p>




## Update Log:  

* [2025.01.08] **Activating Associative Disease-Aware Vision Token Memory for LLM-Based X-ray Report Generation**, arXiv:2501.03458 is released on
  [[arXiv](https://arxiv.org/abs/2501.03458)] 
  
* [2024.10.23] **Pre-training on High Definition X-ray Images: An Experimental Study** is accepted by [**Visual Intelligence (VI)**](https://link.springer.com/journal/44267) Journal. 

* [2024.10.01] **A Pre-trained Large Model MambaXray-VL and the benchmark for the CheXpert Plus dataset** is released [arXiv:2410.00379]

* [2024.08.20] **R2GenCSR: Retrieving Context Samples for Large Language Model based X-ray Medical Report Generation** is released [arXiv:2408.09743]

* [2024.04.28] **Pre-training on High Definition X-ray Images: An Experimental Study**, Xiao Wang, Yuehang Li, Wentao Wu, Jiandong Jin, Yao Rong, Bo Jiang, Chuanfu Li, Jin Tang
arXiv pre-print, arXiv 2024, is released on **arXiv** [[Paper](https://arxiv.org/abs/2404.17926)]


## Surveys/Reviews 
* **Automated Radiology Report Generation: A Review of Recent Advances**, Phillip Sloan, Philip Clatworthy, Edwin Simpson, Majid Mirmehdi
  [[Paper](https://arxiv.org/abs/2405.10842)]


## Projects Maintained in This Github:  


### :dart: [Activating Associative Disease-Aware Vision Token Memory for LLM-Based X-ray Report Generation]()   
Xiao Wang, Fuling Wang, Haowen Wang, Bo Jiang, Chuanfu Li, Yaowei Wang, Yonghong Tian, Jin Tang, 
arXiv Pre-print arXiv:2501.03458, 2025 
[[Paper](https://arxiv.org/abs/2501.03458)]

X-ray image based medical report generation achieves significant progress in recent years with the help of the large language model, however, these models have not fully exploited the effective information in visual image regions, resulting in reports that are linguistically sound but insufficient in describing key diseases. In this paper, we propose a novel associative memory-enhanced X-ray report generation model that effectively mimics the process of professional doctors writing medical reports. It considers both the mining of global and local visual information and associates historical report information to better complete the writing of the current report. Specifically, given an X-ray image, we first utilize a classification model along with its activation maps to accomplish the mining of visual regions highly associated with diseases and the learning of disease query tokens. Then, we employ a visual Hopfield network to establish memory associations for disease-related tokens, and a report Hopfield network to retrieve report memory information. This process facilitates the generation of high-quality reports based on a large language model and achieves state-of-the-art performance on multiple benchmark datasets, including the IU X-ray, MIMIC-CXR, and Chexpert Plus.

![CXPMRG_Bench](https://github.com/Event-AHU/Medical_Image_Analysis/blob/main/AM_MRG/figures/framework_FulingWang.jpg)




### :dart: [A Pre-trained Large Model MambaXray-VL and the benchmark for the CheXpert Plus dataset]()  
* **CXPMRG-Bench: Pre-training and Benchmarking for X-ray Medical Report Generation on CheXpert Plus Dataset**, 
Xiao Wang, Fuling Wang, Yuehang Li, Qingchuan Ma, Shiao Wang, Bo Jiang, Chuanfu Li, Jin Tang, 
arXiv Pre-print arXiv:2410.00379, 2024
[[Paper](https://arxiv.org/abs/2410.00379)]

X-ray image-based medical report generation (MRG) is a pivotal area in artificial intelligence which can significantly reduce diagnostic burdens and patient wait times. Despite significant progress, we believe that the task has reached a bottleneck due to the limited benchmark datasets and the existing large models' insufficient capability enhancements in this specialized domain. Specifically, the recently released CheXpert Plus dataset lacks comparative evaluation algorithms and their results, providing only the dataset itself. This situation makes the training, evaluation, and comparison of subsequent algorithms challenging. Thus, we conduct a comprehensive benchmarking of existing mainstream X-ray report generation models and large language models (LLMs), on the CheXpert Plus dataset. We believe that the proposed benchmark can provide a solid comparative basis for subsequent algorithms and serve as a guide for researchers to quickly grasp the state-of-the-art models in this field. More importantly, we propose a large model for the X-ray image report generation using a multi-stage pre-training strategy, including self-supervised autoregressive generation and Xray-report contrastive learning, and supervised fine-tuning. Extensive experimental results indicate that the autoregressive pre-training based on Mamba effectively encodes X-ray images, and the image-text contrastive pre-training further aligns the feature spaces, achieving better experimental results. 

![CXPMRG_Bench](https://github.com/Event-AHU/Medical_Image_Analysis/blob/main/CXPMRG_Bench_MambaXray_VL/CXPMRG_Bench.png)
![CXPMRG_Bench](https://github.com/Event-AHU/Medical_Image_Analysis/blob/main/CXPMRG_Bench_MambaXray_VL/MambaXray_VL.jpg)



### :dart: [Context Sample Retrieval for LLM based X-ray Report Generation]()  
* **R2GenCSR: Retrieving Context Samples for Large Language Model based X-ray Medical Report Generation**, 
Xiao Wang, Yuehang Li, Fuling Wang, Shiao Wang, Chuanfu Li, Bo Jiang, 
arXiv Pre-print arXiv:2408.09743, 2024 
[[Paper](https://arxiv.org/abs/2408.09743)]

Inspired by the tremendous success of Large Language Models (LLMs), existing X-ray medical report generation methods attempt to leverage large models to achieve better performance. They usually adopt a Transformer to extract the visual features of a given X-ray image, and then, feed them into the LLM for text generation. How to extract more effective information for the LLMs to help them improve final results is an urgent problem that needs to be solved. Additionally, the use of visual Transformer models also brings high computational complexity. To address these issues, this paper proposes a novel context-guided efficient X-ray medical report generation framework. Specifically, we introduce the Mamba as the vision backbone with linear complexity, and the performance obtained is comparable to that of the strong Transformer model. More importantly, we perform context retrieval from the training set for samples within each mini-batch during the training phase, utilizing both positively and negatively related samples to enhance feature representation and discriminative learning. Subsequently, we feed the vision tokens, context information, and prompt statements to invoke the LLM for generating high-quality medical reports. Extensive experiments on three X-ray report generation datasets (i.e., IU-Xray, MIMIC-CXR, CheXpert Plus) fully validated the effectiveness of our proposed model. 

![R2GenCSR](https://github.com/Event-AHU/Medical_Image_Analysis/blob/main/R2GenCSR/framework.jpg)



### :dart: [Pre-training MAE Model on HD X-ray Images]() 
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
  

## :newspaper: Citation 
If you find this work helps your research, please star this GitHub and cite the following papers: 
```bibtex
@misc{wang2025AMMRG,
      title={Activating Associative Disease-Aware Vision Token Memory for LLM-Based X-ray Report Generation}, 
      author={Xiao Wang and Fuling Wang and Haowen Wang and Bo Jiang and Chuanfu Li and Yaowei Wang and Yonghong Tian and Jin Tang},
      year={2025},
      eprint={2501.03458},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2501.03458}, 
}

@misc{wang2024CXPMRGBench,
      title={CXPMRG-Bench: Pre-training and Benchmarking for X-ray Medical Report Generation on CheXpert Plus Dataset}, 
      author={Xiao Wang and Fuling Wang and Yuehang Li and Qingchuan Ma and Shiao Wang and Bo Jiang and Chuanfu Li and Jin Tang},
      year={2024},
      eprint={2410.00379},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.00379}, 
}

@misc{wang2024R2GenCSR,
      title={R2GenCSR: Retrieving Context Samples for Large Language Model based X-ray Medical Report Generation}, 
      author={Xiao Wang and Yuehang Li and Fuling Wang and Shiao Wang and Chuanfu Li and Bo Jiang},
      year={2024},
      eprint={2408.09743},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.09743}, 
}

@misc{wang2024pretraininghighdefinitionxray,
      title={Pre-training on High Definition X-ray Images: An Experimental Study}, 
      author={Xiao Wang and Yuehang Li and Wentao Wu and Jiandong Jin and Yao Rong and Bo Jiang and Chuanfu Li and Jin Tang},
      year={2024},
      eprint={2404.17926},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2404.17926}, 
}

```

If you have any questions about these works, please feel free to leave an issue. 


## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=Event-AHU/Medical_Image_Analysis&type=Date)](https://star-history.com/#Event-AHU/Medical_Image_Analysis&Date)







  
