# Initialize model and only move data to GPU as needed
from transformers import BertModel
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

class DataLoader4BERT:
    def __init__(self, gdsc_path, compounds_path, gdsc2_path, cell_lines_path, bert_model_name="bert-base-uncased"):
        self.gdsc_path = gdsc_path
        self.compounds_path = compounds_path
        self.gdsc2_path = gdsc2_path
        self.cell_lines_path = cell_lines_path
        self.bert_model_name = bert_model_name
        
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        
        self.gdsc_dataset = None
        self.compounds_annotation = None
        self.gdsc2_dataset = None
        self.cell_lines_details = None
        self.final_df = None
        self.X_train_text = None
        self.X_test_text = None
        self.X_train_numeric = None
        self.X_test_numeric = None
        self.y_train = None
        self.y_test = None
        
        self.load_data()
        self.preprocess_data()
        self.define_features_and_target()

    def load_data(self):
        self.gdsc_dataset = pd.read_csv(self.gdsc_path)
        self.compounds_annotation = pd.read_csv(self.compounds_path)
        self.gdsc2_dataset = pd.read_csv(self.gdsc2_path)
        self.cell_lines_details = pd.read_excel(self.cell_lines_path)
        print('Loading Done!')

    def preprocess_data(self):
        self.gdsc_dataset = self.gdsc_dataset.dropna()
        self.compounds_annotation = self.compounds_annotation.dropna()
        self.gdsc2_dataset = self.gdsc2_dataset.dropna()
        self.cell_lines_details = self.cell_lines_details.dropna()

        merged_df = pd.merge(self.gdsc2_dataset, self.cell_lines_details, left_on='COSMIC_ID', right_on='COSMIC identifier', how='left')
        self.final_df = pd.merge(merged_df, self.compounds_annotation, on='DRUG_ID', how='left')
        print('Preprocess Done!')

    def define_features_and_target(self):
        numeric_features = ['AUC', 'Z_SCORE']
        text_features = ['Cancer Type\n(matching TCGA label)', 
                        'GDSC\nTissue descriptor 1', 
                        'GDSC\nTissue\ndescriptor 2']

        for feature in numeric_features:
            if feature not in self.final_df.columns:
                raise ValueError(f"數值特徵 {feature} 不存在於 final_df 中。")

        X_numeric = self.final_df[numeric_features].fillna(0).astype(float)
        if X_numeric.empty:
            raise ValueError("數值特徵提取結果為空，請檢查數據處理步驟。")

        X_text = self.final_df[text_features].fillna('')
        text_inputs = X_text.apply(lambda x: ' '.join(x), axis=1).tolist()
        tokenized = self.tokenizer(text_inputs, padding=True, truncation=True, return_tensors="pt", max_length=16)

        if 'LN_IC50' not in self.final_df.columns:
            raise ValueError("目標變數 'LN_IC50' 不存在於 final_df 中。")
        y = self.final_df['LN_IC50']

        X_train_numeric, X_test_numeric, y_train, y_test = train_test_split(
            X_numeric, y, test_size=0.2, random_state=42
        )
        input_ids_train, input_ids_test, attention_mask_train, attention_mask_test = train_test_split(
            tokenized['input_ids'], tokenized['attention_mask'], test_size=0.2, random_state=42
        )

        self.X_train_numeric, self.X_test_numeric = X_train_numeric, X_test_numeric
        self.y_train, self.y_test = y_train, y_test
        self.X_train_text = {"input_ids": input_ids_train, "attention_mask": attention_mask_train}
        self.X_test_text = {"input_ids": input_ids_test, "attention_mask": attention_mask_test}

        print("數值特徵與文本特徵分配完成")

    def get_data(self):
        # Return tensors in CPU initially
        X_train_numeric_tensor = torch.tensor(self.X_train_numeric.values).float()
        X_test_numeric_tensor = torch.tensor(self.X_test_numeric.values).float()
        y_train_tensor = torch.tensor(self.y_train.values).float()
        y_test_tensor = torch.tensor(self.y_test.values).float()
        
        X_train_text = self.X_train_text
        X_test_text = self.X_test_text

        return X_train_numeric_tensor, X_train_text, y_train_tensor, X_test_numeric_tensor, X_test_text, y_test_tensor

def train(model, input_ids, attention_mask, numeric_features, targets, optimizer, batch_size=16):
    model.train()
    total_loss = 0
    for i in range(0, input_ids.size(0), batch_size):
        batch_input_ids = input_ids[i:i + batch_size]
        batch_attention_mask = attention_mask[i:i + batch_size]
        batch_numeric_features = numeric_features[i:i + batch_size]
        batch_targets = targets[i:i + batch_size].to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_input_ids, batch_attention_mask, batch_numeric_features)
        # loss = F.l1_loss(outputs.squeeze(), batch_targets)
        loss = F.mse_loss(outputs.squeeze(), batch_targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / (input_ids.size(0) // batch_size)

def evaluate(model, input_ids, attention_mask, numeric_features, targets, batch_size=16):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for i in range(0, input_ids.size(0), batch_size):
            batch_input_ids = input_ids[i:i + batch_size]
            batch_attention_mask = attention_mask[i:i + batch_size]
            batch_numeric_features = numeric_features[i:i + batch_size]
            batch_targets = targets[i:i + batch_size].to(device)
            
            outputs = model(batch_input_ids, batch_attention_mask, batch_numeric_features)
            predictions.extend(outputs.squeeze().cpu().numpy())
            actuals.extend(batch_targets.cpu().numpy())
    
    mae = mean_absolute_error(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)
    
    return mae, mse

class BertForNumericPrediction(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased", num_numeric_features=2):
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
        return output

# Load dataset and model
import kagglehub

path = kagglehub.dataset_download("samiraalipour/genomics-of-drug-sensitivity-in-cancer-gdsc")
print("Path to dataset files:", path)

dataloader = DataLoader4BERT(path + '/GDSC_DATASET.csv',
                             path + '/Compounds-annotation.csv',
                             path + '/GDSC2-dataset.csv',
                             path + '/Cell_Lines_Details.xlsx')

X_train_numeric, X_train_text, y_train_tensor, X_test_numeric, X_test_text, y_test_tensor = dataloader.get_data()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move necessary data to GPU
input_ids_train = X_train_text["input_ids"].to(device)
attention_mask_train = X_train_text["attention_mask"].to(device)
X_train_numeric = X_train_numeric.to(device)
y_train_tensor = y_train_tensor.to(device)

input_ids_test = X_test_text["input_ids"].to(device)
attention_mask_test = X_test_text["attention_mask"].to(device)
X_test_numeric = X_test_numeric.to(device)
y_test_tensor = y_test_tensor.to(device)

model = BertForNumericPrediction(bert_model_name="bert-base-uncased", num_numeric_features=X_train_numeric.shape[1])
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=2e-5)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    train_loss = train(model, input_ids_train, attention_mask_train, X_train_numeric, y_train_tensor, optimizer)
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}")

# Evaluate on test data
mae, mse = evaluate(model, input_ids_test, attention_mask_test, X_test_numeric, y_test_tensor)
print(f"Mean Absolute Error on Test Data: {mae:.4f}")
print(f"Mean Square Error on Test Data: {mse:.4f}")
