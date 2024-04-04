from torch.utils.data import Dataset
import numpy as np
import io
from PIL import Image
import os
import json
import random
from image_synthesis.utils.misc import instantiate_from_config
import torch

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

class CocoDataset(Dataset):
    def __init__(self, data_root, phase = 'train', im_preprocessor_config=None, drop_caption_rate=0.0, tokenize_config=None):
        if tokenize_config is not None:
            self.tokenizer = instantiate_from_config(tokenize_config)
        self.transform = instantiate_from_config(im_preprocessor_config)
        self.root = os.path.join(data_root, phase)
        # input_file = os.path.join(data_root, input_file)
        caption_file = "captions_"+phase+"2014.json"
        caption_file = os.path.join(data_root, "annotations", caption_file)

        self.json_file = json.load(open(caption_file, 'r'))
        print("length of the dataset is ")
        print(len(self.json_file['annotations']))

        self.num = len(self.json_file['annotations'])
        self.image_prename = "COCO_" + phase + "2014_"
        self.folder_path = os.path.join(data_root, phase+'2014', phase+'2014')
 
        self.drop_rate = drop_caption_rate
        self.phase = phase
 
    def __len__(self):
        return self.num
 
    def __getitem__(self, index):
        this_item = self.json_file['annotations'][index]
        caption = this_item['caption'].lower()
        if hasattr(self, "tokenizer"):
            caption = self.tokenizer.get_tokens(caption)
        image_name = str(this_item['image_id']).zfill(12)
        image_path = os.path.join(self.folder_path, self.image_prename+image_name+'.jpg')
        image = load_img(image_path)
        image = np.array(image).astype(np.uint8)
        image = self.transform(image = image)['image']
        data = {
                'image': np.transpose(image.astype(np.float32), (2, 0, 1)),
                'text': caption if (self.phase != 'train' or self.drop_rate < 1e-6 or random.random() >= self.drop_rate) else '',
        }
        return data
    
class CocoCaptionDataset(Dataset):
    def __init__(self, data_root, tokenize_config=None):
        if tokenize_config is not None:
            self.tokenizer = instantiate_from_config(tokenize_config)
        self.data_root = data_root
        self.caption_list = self.get_list()
            
    def get_list(self):
        with open(self.data_root, 'r') as caption_file:
            caption = caption_file.readlines()
        caption = list(map(lambda x:x.strip(), caption))
        return caption 
    
    def __len__(self):
        return len(self.caption_list)
    
    def __getitem__(self, index):
        caption = self.caption_list[index]
        return {'text':self.tokenizer.get_tokens(caption), 
                'text_origin':self.caption_list[index]}


def CocoConcatDataset(data_root, 
                    phase_first='train', 
                    im_preprocessor_config_first=None, 
                    drop_caption_rate_first=0.0, 
                    phase_second='valid',
                    im_preprocessor_config_second=None, 
                    drop_caption_rate_second=0.0):
    first_coco = CocoDataset(data_root=data_root, 
                             phase=phase_first, 
                             im_preprocessor_config=im_preprocessor_config_first, 
                             drop_caption_rate=drop_caption_rate_first)
    second_coco = CocoDataset(data_root=data_root, 
                             phase=phase_second, 
                             im_preprocessor_config=im_preprocessor_config_second, 
                             drop_caption_rate=drop_caption_rate_second)
    return torch.utils.data.ConcatDataset([first_coco, second_coco])