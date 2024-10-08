import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import os
import datasets
import argparse
from typing import Tuple
import transformers
import torch
from torch.utils.data import Dataset
import matplotlib as plt
import random
from tqdm import tqdm
import pandas as pd
from huggingface_hub import login
from torch.optim import lr_scheduler
from typing import Callable, Dict, List, Tuple, Union
import csv
from timeit import default_timer as timer





def load_tokenizer(tokenizer_name:str)->object:
    """
    Function to load the tokenizer by the model's name
    Args: 
     - tokenizer_name -> the name of the tokenizerto download
     Returns:
     - tokenizer -> returns respectively the model and the tokenizer
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained("Salesforce/codet5p-770m")


    return tokenizer


def load_model(model_name:str)->object:
    """
     Function for model loading
     Args: 
     - model_name -> the name of the model
     Returns:
     - model,tokenizer -> returns respectively the model and the tokenizer
    """

    print(f'Loading  model {model_name}...')


    model_kwargs = {}

    model_kwargs.update(dict( torch_dtype=torch.bfloat16))
    transformers.T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
    model_encoder = transformers.T5EncoderModel.from_pretrained("Salesforce/codet5p-770m", **model_kwargs)

    print("---MODEL LOADED---")

   

    return model_encoder

class stylometer_classifier(torch.nn.Module):
    def __init__(self,pretrained_encoder,dimensionality):
        super(stylometer_classifier, self).__init__()
        self.modelBase = pretrained_encoder
        self.pre_classifier = torch.nn.Linear(dimensionality, 768, dtype=torch.bfloat16)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)
        self.classifier = torch.nn.Linear(768, 1, dtype=torch.bfloat16)




    def forward(self, input_ids, padding_mask):
        output_1 = self.modelBase(input_ids=input_ids, attention_mask=padding_mask)
        hidden_state = output_1[0]
        #Here i take only the cls token representation for further classification
        cls_output = hidden_state[:, 0]
        pooler = self.pre_classifier(cls_output)
        afterActivation = self.activation(pooler)
        pooler_after_act = self.dropout(afterActivation)
        output = torch.sigmoid(self.classifier(pooler_after_act))

        if output>=0.07:
            return {"my_class":"It's a Human!",
                   "prob":output}
        else:
            return {"my_class":"It's an LLM!",
                   "prob":output}


        return output

def adapt_model(model:object, dim:int=1024) -> object:
    """
    This function returns the model with a classification head
    """
    newModel = stylometer_classifier(model,dimensionality=dim)

    return newModel





def main():
    print("----starting enviroment----")


    model_name = "Salesforce/codet5p-770m"
    checkpoint = "checkpoint.bin"


    DEVICE = "cpu"



    #load tokenizer
    tokenizer = load_tokenizer(model_name)
    print("tokenizer  loaded!")
 

    #loading model and tokenizer for functional translation
    model = load_model(model_name)
    #adding classification head to the model
    model = adapt_model(model, dim=model.shared.embedding_dim)



    model.load_state_dict(torch.load(checkpoint,map_location='cpu'))
    model = model.eval()
    st.title("Human-AI stylometer - Multilingual")
    
    st.caption('Is This You, LLM? Recognizing AI-written Programs with Multilingual Code Stylometry')
    
    text = st.text_area("insert your code here")
    button = st.button("send")
    if button or text:
        input = tokenizer([text])
        out= model(torch.tensor(input.input_ids),torch.tensor(input.attention_mask))
        st.write(out["my_class"]) 
    



if __name__ == '__main__':
    main()