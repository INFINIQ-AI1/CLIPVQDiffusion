import torch
import math
from torch import nn
from image_synthesis.utils.misc import instantiate_from_config
import time
import numpy as np
from PIL import Image
import os
from torchvision import transforms
import torchvision
from torch.cuda.amp import autocast
from einops import einsum 

def cosine_metric(x1, x2):
    #return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
    return torch.sum(x1 * x2, dim=1) / (torch.norm(x1, dim=1) * torch.norm(x2, dim=1))

class CLIPVQDiffusion(nn.Module):
    def __init__(
        self,
        *,
        content_info={'key': 'image'},
        condition_info_train={'key': 'label'},
        condition_info_valid={'key': 'label'},
        guidance_scale=1.0,
        learnable_cf=False,
        use_lafite=False, 
        lafite_alpha=0.9, 
        lafite_norm=True,
        times=16,
        truncation_r="0.85",
        tokenizer_config=None,
        guidance_text="",
        text_guidance_alpha=1,  
        content_codec_config,
        condition_codec_config, 
        diffusion_config, 
    ):
        super().__init__()
        self.content_info = content_info
        self.condition_info_train = condition_info_train
        self.condition_info_valid = condition_info_valid
        self.learnable_cf = learnable_cf 
        self.guidance_scale = guidance_scale
        self.condition_codec = instantiate_from_config(condition_codec_config)
        self.transformer = instantiate_from_config(diffusion_config)
        self.content_codec = instantiate_from_config(content_codec_config)
        self.truncation_forward = False
        self.use_lafite=use_lafite
        self.lafite_alpha=lafite_alpha
        self.lafite_norm=lafite_norm
        self.clip_img_preprocess = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        self.times=times
        self.truncation_r=truncation_r
            
        
    def parameters(self, recurse=True, name=None):
        # return super().parameters(recurse=True)
        if name is None or name == 'none':
            return super().parameters(recurse=recurse)
        else:
            names = name.split('+')
            params = []
            for n in names:
                try: # the parameters() method is not overwritten for some classes
                    params += getattr(self, name).parameters(recurse=recurse, name=name)
                except:
                    params += getattr(self, name).parameters(recurse=recurse)
            return params

    @property
    def device(self):
        return self.transformer.device

    def get_ema_model(self):
        return self.transformer

    @torch.no_grad()
    def prepare_condition(self, batch, desc="train"):
        if desc=="train":
            cond_key = self.condition_info_train['key']
        elif desc=="valid":
            cond_key = self.condition_info_valid['key']
        else:
            raise Exception("desc should be train or valid")
        cond = batch[cond_key]
        if torch.is_tensor(cond):
            cond = cond.to(self.device)
        cond_ = {}
        if desc=="train":
            cond_['condition_embed_token'] = self.condition_codec.encode_image(self.clip_img_preprocess(transforms.Resize(224)(cond/255)))
            if self.use_lafite:
                randn = torch.randn(cond_["condition_embed_token"].shape).to(self.device)    
                cond_['condition_embed_token'] = cond_['condition_embed_token'] + self.lafite_alpha*(randn/(randn.norm(dim=-1, keepdim=True)))
                if self.lafite_norm:
                    cond_['condition_embed_token'] = cond_['condition_embed_token']/(cond_['condition_embed_token'].norm(dim=-1, keepdim=True))
        else: 
            cond_['condition_embed_token'] = self.condition_codec.encode_text(cond['token'])
        return cond_

    @autocast(enabled=False)
    @torch.no_grad()
    def prepare_content(self, batch, with_mask=False):
        cont_key = self.content_info['key']
        cont = batch[cont_key]
        if torch.is_tensor(cont):
            cont = cont.to(self.device)
        if not with_mask:
            cont = self.content_codec.get_tokens(cont)
        else:
            mask = batch['mask'.format(cont_key)]
            cont = self.content_codec.get_tokens(cont, mask, enc_with_mask=False)
        cont_ = {}
        for k, v in cont.items():
            v = v.to(self.device) if torch.is_tensor(v) else v
            cont_['content_' + k] = v
        return cont_
    
    @torch.no_grad()
    def prepare_input(self, batch):
        input = self.prepare_condition(batch, desc="train")
        input.update(self.prepare_content(batch))
        return input

    def predict_start_with_truncation(self, func, sample_type):
        if sample_type[-1] == 'p':
            truncation_k = int(sample_type[:-1].replace('top', ''))
            content_codec = self.content_codec
            save_path = self.this_save_path
            def wrapper(*args, **kwards):
                out = func(*args, **kwards)
                val, ind = out.topk(k = truncation_k, dim=1)
                probs = torch.full_like(out, -70)
                probs.scatter_(1, ind, val)
                return probs
            return wrapper
        elif sample_type[-1] == 'r':
            truncation_r = float(sample_type[:-1].replace('top', ''))
            def wrapper(*args, **kwards):
                out = func(*args, **kwards)
                temp, indices = torch.sort(out, 1, descending=True)
                temp1 = torch.exp(temp)
                temp2 = temp1.cumsum(dim=1)
                temp3 = temp2 < truncation_r
                new_temp = torch.full_like(temp3[:,0:1,:], True)
                temp6 = torch.cat((new_temp, temp3), dim=1)
                temp3 = temp6[:,:-1,:]
                temp4 = temp3.gather(1, indices.argsort(1))
                temp5 = temp4.float()*out+(1-temp4.float())*(-70)
                probs = temp5
                return probs
            return wrapper
        else:
            print("wrong sample type")


    @torch.no_grad()
    def generate_content(
        self,
        *,
        batch,
        condition=None,
        filter_ratio = 0.5,
        temperature = 1.0,
        content_ratio = 0.0,
        replicate=1,
        return_att_weight=False,
        sample_type="normal",
    ):
        self.eval()
        if condition is None:
            condition = self.prepare_condition(batch=batch, desc="valid")
        else:
            condition = self.prepare_condition(batch=None, condition=condition, desc="valid")
        
        kwargs={}
        if replicate != 1:
            kwargs["batch_size"] = replicate
        else:
            kwargs["batch_size"] = condition["condition_embed_token"].shape[0]
        # content = None

        if replicate != 1:
            for k in condition.keys():
                if condition[k] is not None:
                    condition[k] = torch.cat([condition[k].unsqueeze(0) for _ in range(replicate)], dim=0)
        
        # times x sample x dim 
        content_token = None

        if self.learnable_cf:
            cf_cond_emb = self.transformer.empty_text_embed.repeat(kwargs["batch_size"], 1)
        else:
            cf_cond_emb = None
        
        guidance_scale = self.guidance_scale

        def cf_predict_start(log_x_t, cond_emb, t):
            log_x_recon = self.transformer.predict_start(log_x_t, cond_emb, t)[:, :-1]
            if abs(guidance_scale - 1) < 1e-3:
                return torch.cat((log_x_recon, self.transformer.zero_vector), dim=1)
            cf_log_x_recon = self.transformer.predict_start(log_x_t, cf_cond_emb.type_as(cond_emb), t)[:, :-1]
            log_new_x_recon = cf_log_x_recon + guidance_scale * (log_x_recon - cf_log_x_recon)
            log_new_x_recon -= torch.logsumexp(log_new_x_recon, dim=1, keepdim=True)
            log_new_x_recon = log_new_x_recon.clamp(-70, 0)
            log_pred = torch.cat((log_new_x_recon, self.transformer.zero_vector), dim=1)
            return log_pred
        if sample_type.split(',')[0][:3] == "top":
            self.transformer.cf_predict_start = self.predict_start_with_truncation(cf_predict_start, sample_type.split(',')[0])
            # self.truncation_forward = True
        
        max_sample_list = []
        if replicate != 1:
            for text_num in range(condition["condition_embed_token"].shape[1]):
                trans_out = self.transformer.sample(condition_token=None,
                                                    condition_mask=None,
                                                    condition_embed=condition["condition_embed_token"][:,text_num,:],
                                                    content_token=content_token,
                                                    filter_ratio=filter_ratio,
                                                    temperature=temperature,
                                                    return_att_weight=return_att_weight,
                                                    return_logits=False,
                                                    print_log=False,
                                                    sample_type=sample_type, 
                                                    **kwargs)
                clip_text_embedding = condition["condition_embed_token"][:,text_num,:]
                sample = self.content_codec.decode(trans_out['content_token'])  #(8,1024)->(8,3,256,256)
                clip_img_embedding = self.condition_codec.encode_image(self.clip_img_preprocess(torchvision.transforms.Resize(224)((sample/255))))
                cos_sim = cosine_metric(clip_text_embedding, clip_img_embedding)
                max_index = cos_sim.argmax(dim=0)
                max_sample = sample[max_index, :]
                max_sample_list.append(max_sample.unsqueeze(0))
            content = torch.concat(max_sample_list, dim=0)
        else:
            trans_out = self.transformer.sample(condition_token=None,
                                                    condition_mask=None,
                                                    condition_embed=condition["condition_embed_token"],
                                                    content_token=content_token,
                                                    filter_ratio=filter_ratio,
                                                    temperature=temperature,
                                                    return_att_weight=return_att_weight,
                                                    return_logits=False,
                                                    print_log=False,
                                                    sample_type=sample_type, 
                                                    **kwargs)
            content = self.content_codec.decode(trans_out['content_token'])  #(8,1024)->(8,3,256,256)
        
        # change cf predict start 
        self.transformer.cf_predict_start = self.transformer.predict_start 
        self.train()
        out = {
            'content': content
        }
        return out

    @torch.no_grad()
    def reconstruct(
        self,
        input
    ):
        if torch.is_tensor(input):
            input = input.to(self.device)
        cont = self.content_codec.get_tokens(input)
        cont_ = {}
        for k, v in cont.items():
            v = v.to(self.device) if torch.is_tensor(v) else v
            cont_['content_' + k] = v
        rec = self.content_codec.decode(cont_['content_token'])
        return rec

    @torch.no_grad()
    def sample(
        self,
        batch,
        clip = None,
        temperature = 1.,
        return_rec = True,
        filter_ratio = [0, 0.5, 1.0],
        content_ratio = [1], # the ratio to keep the encoded content tokens
        return_att_weight=False,
        return_logits=False,
        sample_type="normal",
        phase="train",
        **kwargs,
    ):
        self.eval()
        if phase == "train":
            condition = self.prepare_condition(batch, desc="train")
            content = self.prepare_content(batch)
            kwargs["batch_size"] = condition['condition_embed_token'].shape[0]

            # import pdb; pdb.set_trace()
            if phase == "train":
                content_samples = {'input_image': batch[self.content_info['key']]}
                if return_rec:
                    content_samples['reconstruction_image'] = self.content_codec.decode(content['content_token'])  
                for fr in filter_ratio:
                    for cr in content_ratio:
                        num_content_tokens = int((content['content_token'].shape[1] * cr))
                        if num_content_tokens < 0:
                            continue
                        else:
                            content_token = content['content_token'][:, :num_content_tokens]
                        trans_out = self.transformer.sample(condition_token=condition.get('condition_token', None),
                                                            condition_mask=condition.get('condition_mask', None),
                                                            condition_embed=condition.get('condition_embed_token', None),
                                                            content_token=content_token,
                                                            filter_ratio=fr,
                                                            temperature=temperature,
                                                            return_att_weight=return_att_weight,
                                                            return_logits=return_logits,
                                                            content_logits=content.get('content_logits', None),
                                                            sample_type=sample_type,
                                                            **kwargs)

                        content_samples['cond1_cont{}_fr{}_image'.format(cr, fr)] = self.content_codec.decode(trans_out['content_token'])

                        if return_att_weight:
                            content_samples['cond1_cont{}_fr{}_image_condition_attention'.format(cr, fr)] = trans_out['condition_attention'] # B x Lt x Ld
                            content_att = trans_out['content_attention']
                            shape = *content_att.shape[:-1], self.content.token_shape[0], self.content.token_shape[1]
                            content_samples['cond1_cont{}_fr{}_image_content_attention'.format(cr, fr)] = content_att.view(*shape) # B x Lt x Lt -> B x Lt x H x W
                        if return_logits:
                            content_samples['logits'] = trans_out['logits']
                self.train() 
                output = {'condition': batch[self.condition_info_train['key']]}   
                output.update(content_samples)
                return output
        elif phase == "valid":
            # use generate content 
            # change prior_rule, prior_weight in the transformer itself 
            model_out = self.generate_content(batch=batch, 
                                              filter_ratio=0, 
                                              replicate=self.times, 
                                              content_ratio=1, 
                                              return_att_weight=False, 
                                              sample_type="top"+str(self.truncation_r)+"r")
            self.train() 
            return model_out["content"]

    def forward(
        self,
        batch,
        name='none',
        **kwargs
    ):
        input = self.prepare_input(batch)
        output = self.transformer(input, **kwargs)
        return output
