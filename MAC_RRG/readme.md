## 

**MAC-RRG: Iterative Multi-Agent Collaboration for X-ray Radiology Report Generation**, 
Futian Wang, Yuhan Qiao, Xiao Wang*, Dan Xu, Yuehang Li, Zhixiang Guo*, Jin Tang 


#### **Abstract** 
Despite the remarkable progress of LLM-based and knowledge graph-augmented Radiology Report Generation (RRG) methods, existing techniques still suffer from inherent defects. Conventional LLM-only models lack structured medical prior knowledge, resulting in frequent medical hallucinations and low diagnostic interpretability. Current knowledge graph-enhanced schemes adopt static one-round knowledge fusion with single-source knowledge, incapable of dynamic knowledge updating according to generation feedback. This paper proposes a novel multi-agent collaborative iterative framework for X-ray radiology report generation, termed MAC-RRG. Inspired by multi-agent technology, our framework constructs a closed-loop optimization paradigm based on task decoupling and collaborative reasoning. Specifically, the framework first generates a preliminary radiology report from input X-ray images via a vision encoder and a basic LLM. Subsequently, a multimodal knowledge graph (MM-KG) agent mines structured disease correlation and anatomical knowledge from medical knowledge graphs, while an auxiliary knowledge agent extracts unstructured domain knowledge from public medical databases. The multi-source knowledge acquired by dual agents is fused and embedded to guide the LLM in iteratively refining the initial report. Extensive quantitative and qualitative experiments on mainstream X-ray RRG datasets verify the superiority of our proposed method. 

#### Framework 


#### Environment Configuration 


#### Training and Testing 


#### Experimental Results 


#### Acknowledgment 



#### Citation 
If you find this work useful for your research, please give us a star and cite the following paper. 
```
MAC-RRG: Iterative Multi-Agent Collaboration for X-ray Radiology Report Generation
```


