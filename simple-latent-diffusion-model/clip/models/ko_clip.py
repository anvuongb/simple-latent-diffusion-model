import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoProcessor, AutoModel, AutoTokenizer

class KoCLIPWrapper(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.model_name = "koclip/koclip-base-pt"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
    def loss(self, image, text):
        image_features, text_features = self(image, text, tokenize=False)

        # Normalize features
        image_features = F.normalize(image_features, dim=1)
        text_features = F.normalize(text_features, dim=1)

        # Cosine similarity as logits with learned temperature
        logits = torch.matmul(image_features, text_features.t()) * self.logit_scale.exp()
        labels = torch.arange(logits.shape[0], dtype=torch.long, device=logits.device)

        # Cross-entropy loss
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)

        return (loss_i2t + loss_t2i) / 2

    def text_encode(self, text, tokenize):
        if tokenize:
            tokens = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")['input_ids']
        else:
            tokens = text
        tokens = tokens.to(self.model.device)
        return self.model.get_text_features(tokens)
    
    def forward(self, image, text, tokenize=True):
        if tokenize==False:
            text = self.tokenizer.decode(text, skip_special_tokens=True)
        inputs = self.processor(
            text=text,
            images=image, 
            return_tensors="pt",
            )
        outputs = self.model(**inputs)
            
        return outputs.image_embeds, outputs.text_embeds # [1, 512], [1, 512]