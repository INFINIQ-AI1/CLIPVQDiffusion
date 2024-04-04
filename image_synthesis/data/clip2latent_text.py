from torch.utils.data import Dataset
import numpy as np
import io
from PIL import Image
import os
import json
import random
from image_synthesis.utils.misc import instantiate_from_config
from image_synthesis.data.utils.clip2latent_text import clip2latent_text_prompt, clip2latent_text_prompt_random

class Clip2latentTextDataset(Dataset):
    def __init__(self, 
                 data_root=None,
                 tokenize_config=None):
        self.text_list = clip2latent_text_prompt 
        self.num = len(self.text_list)
        self.tokenize = instantiate_from_config(tokenize_config)
        
    def __len__(self):
        return self.num
 
    def __getitem__(self, index):
        pretext = "a photograph of "
        text = pretext + self.text_list[index]
        
        text_token = self.tokenize.get_tokens(text)
        data = {
                'text': text_token, 
                'text_origin': text
        }
        return data


class Clip2latentRandomTextDataset(Dataset):
    def __init__(self, 
                 data_root=None,
                 tokenize_config=None):
        self.text_list = clip2latent_text_prompt_random
        self.num = len(self.text_list)
        self.tokenize = instantiate_from_config(tokenize_config)
        
    def __len__(self):
        return self.num
 
    def __getitem__(self, index):
        pretext = "a photograph of "
        text = pretext + self.text_list[index]
        
        text_token = self.tokenize.get_tokens(text)
        data = {
                'text': text_token
        }
        return data
