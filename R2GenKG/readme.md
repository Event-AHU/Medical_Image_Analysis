# Official PyTorch implementation of R2GenKG 

* **R2GenKG: Hierarchical Multi-modal Knowledge Graph for LLM-based Radiology Report Generation**, 
  Futian Wang, Yuhan Qiao, Xiao Wang*, Fuling Wang, Yuxiang Zhang, Dengdi Sun, 
  arXiv:2508.03426,
  [[Paper](https://arxiv.org/abs/2508.03426)] 

## Abstract 
X-ray medical report generation is one of the important applications of artificial intelligence in healthcare. With the support of large foundation models, the quality of medical report generation has significantly improved. However, challenges such as hallucination and weak disease diagnostic capability still persist. In this paper, we first construct a large-scale multi-modal medical knowledge graph (termed M3KG) based on the ground truth medical report using the GPT-4o. It contains 2477 entities, 3 kinds of relations, 37424 triples, and 6943 disease-aware vision tokens for the CheXpert Plus dataset. Then, we sample it to obtain multi-granularity semantic graphs and use an R-GCN encoder for feature extraction. For the input X-ray image, we adopt the Swin-Transformer to extract the vision features and interact with the knowledge using cross-attention. The vision tokens are fed into a Q-former and retrieved the disease-aware vision tokens using another cross-attention. Finally, we adopt the large language model to map the semantic knowledge graph, input X-ray image, and disease-aware vision tokens into language descriptions. Extensive experiments on multiple datasets fully validated the effectiveness of our proposed knowledge graph and X-ray report generation framework. The source code of this paper will be released. 



## Framework  

![overview](https://github.com/Event-AHU/Medical_Image_Analysis/blob/main/R2GenKG/figures/KG_construction.jpg)
An illustration of the proposed multi-modal medical knowledge graph M3KG.

![overview](https://github.com/Event-AHU/Medical_Image_Analysis/blob/main/R2GenKG/figures/R2GenKG_framework.jpg)
An overview of our proposed hierarchical knowledge graph guided X-ray medical report generation framework, termed R2GenKG.


## Getting Started
### Installation

**1. Install requirements**

Install requirements using pip:

```bash
pip install -r requirements.txt
```


**2. Prepare dataset**

We follow R2Gen dataset process to download [IU-Xray](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view) 
For CheXpert Plus dataset: Create a 'chexpert_plus' folder and download the original CheXpert Plus dataset from [here](https://stanfordaimi.azurewebsites.net/datasets/5158c524-d3ab-4e02-96e9-6ee9efc110a1). You can download our preprocessed annotation file from [here](https://drive.google.com/file/d/1vjh8GXaFQYJXJeLaxLnFtvZxuSZscQd_/view?usp=sharing).



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

Once you have finished the training, you can test the model by running the following method:

For IU-xray:
```bash
bash scripts/1-2.test_iuxray.sh
```

For CheXpert Plus:
```bash
bash scripts/6-2.cheXpert_plus_test.sh
```

**Note**: Complete execution steps, code for processing graph features, and weights will be uploaded later.



### Acknowledgement
* **R2GenGPT** We build our framework based on R2GenKG. 
* **CXPMRG-Bench** The experimental results on the CheXpert Plus dataset are obtained from
  [[CXPMRG-Bench](https://arxiv.org/abs/2410.00379)].


  

### Citation
```
@misc{wang2025r2genkghierarchicalmultimodalknowledge,
      title={R2GenKG: Hierarchical Multi-modal Knowledge Graph for LLM-based Radiology Report Generation}, 
      author={Futian Wang and Yuhan Qiao and Xiao Wang and Fuling Wang and Yuxiang Zhang and Dengdi Sun},
      year={2025},
      eprint={2508.03426},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.03426}, 
}
```


