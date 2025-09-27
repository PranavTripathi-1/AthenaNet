# src/infrastructure/text_encoder.py
import torch
from transformers import DistilBertModel, DistilBertTokenizer

class TextEncoder:
    def __init__(self, model_path=None):
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        if model_path is not None:
            # load saved state_dict
            self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")
            self.model.load_state_dict(torch.load(model_path))
        else:
            # fresh model
            self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.model.eval()

    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # use [CLS] token embedding
        return outputs.last_hidden_state[:,0,:]
