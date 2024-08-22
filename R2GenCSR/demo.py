import argparse
import gradio as gr
from PIL import Image
import os
from configs.config import parser
from dataset.data_module import DataModule
from models.R2GenCSR_domo import R2GenCSR
import lightning.pytorch as pl
import torch
from dataset.data_helper import FieldParser

args = parser.parse_args()
model = R2GenCSR(args)
parser = FieldParser(args)
model.to(dtype=torch.bfloat16).cuda()

def get_report(image):
    print(image)
    # image=Image.fromarray(image)
    samples = {'id':[0],
           'image_path':[image],
           'input_text':['None'],}
    samples = parser.transform_with_parse(samples)
    samples = {'id':[0],
            'image':torch.stack(samples['image']).unsqueeze(0).cuda().bfloat16(),
            'input_text':['None'],}
    outputs =model.demo_test_step(samples,None)
    return outputs[0][0]

inputs=gr.Image(container=False,height=500,type ='filepath')
outputs='text'
title="R2GenCSR"
examples = [
    ['./images/mimic_test0.jpg'],
    ['./images/mimic_test1.jpg'],
    ['./images/mimic_test2.jpg'],
    ['./images/mimic_test3.jpg'],
]
demo = gr.Interface(fn=get_report,examples=examples,
                    title=title,
                    inputs=inputs,
                    outputs=outputs)
demo.queue().launch(server_name="0.0.0.0",server_port=1235,share=False)
