from torch import nn 
import torch 

class NormalClipAdapter(nn.Module):
    def __init__(self, clip=None, model='ViT-B/32', device='cpu', normalize=True):
        super().__init__()
        if clip is not None:
            self.clip = clip 
        else:
            # pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip
            import clip 
            self.clip, _ = clip.load(model, device=device)
        self.clip.requires_grad_(False)
        self.clip.eval()
        self.normalize = normalize 
    
    @torch.no_grad()
    def encode_image(self, image):
        embeds = self.clip.encode_image(image)
        if self.normalize:
            embeds /= embeds.norm(dim=-1, keepdim=True)
        return embeds 
        
    @torch.no_grad()
    def encode_text(self, text):
        embeds = self.clip.encode_text(text)
        if self.normalize:
            embeds /= embeds.norm(dim=-1, keepdim=True)
        return embeds 