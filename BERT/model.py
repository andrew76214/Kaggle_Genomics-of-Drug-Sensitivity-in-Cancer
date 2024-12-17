'''import torch
import torch.nn as nn
from transformers import BertModel

class BertForNumericPrediction(nn.Module):
    def __init__(self, bert_model_name, num_numeric_features=2):
        super(BertForNumericPrediction, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.numeric_embed = nn.Linear(num_numeric_features, self.bert.config.hidden_size)
        self.fc = nn.Linear(self.bert.config.hidden_size * 2, 1)

    def forward(self, input_ids, attention_mask, numeric_features):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = bert_outputs.last_hidden_state[:, 0, :]
        numeric_embedding = self.numeric_embed(numeric_features)
        combined = torch.cat([cls_output, numeric_embedding], dim=-1)
        output = self.fc(combined)
        return output'''
    
    
    
import torch
import torch.nn as nn
from transformers import AutoModel

class TransformerForNumericPrediction(nn.Module):
    def __init__(self, model_name, num_numeric_features=2):
        super(TransformerForNumericPrediction, self).__init__()
        
        self.transformer = AutoModel.from_pretrained(model_name)
        self.model_type = model_name.lower()
        hidden_size = self.transformer.config.hidden_size
        self.numeric_embed = nn.Linear(num_numeric_features, hidden_size)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, input_ids, attention_mask, numeric_features):
        transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = transformer_outputs.last_hidden_state[:, 0, :]
        numeric_embedding = self.numeric_embed(numeric_features)
        combined = torch.cat([cls_output, numeric_embedding], dim=-1)
        output = self.fc(combined)
        return output