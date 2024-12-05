import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import numpy as np

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
        # 數值特徵
        numeric_features = ['AUC', 'Z_SCORE']
        # 文本特徵
        text_features = ['Cancer Type\n(matching TCGA label)', 
                        'GDSC\nTissue descriptor 1', 
                        'GDSC\nTissue\ndescriptor 2']
        
        # 檢查數值特徵是否存在於 DataFrame
        for feature in numeric_features:
            if feature not in self.final_df.columns:
                raise ValueError(f"數值特徵 {feature} 不存在於 final_df 中。")

        # 提取數值和文本特徵
        X_numeric = self.final_df[numeric_features].fillna(0).astype(float)
        if X_numeric.empty:
            raise ValueError("數值特徵提取結果為空，請檢查數據處理步驟。")

        # 文本特徵處理
        X_text = self.final_df[text_features].fillna('')
        text_inputs = X_text.apply(lambda x: ' '.join(x), axis=1).tolist()
        tokenized = self.tokenizer(text_inputs, padding=True, truncation=True, return_tensors="pt", max_length=64)

        # 目標變數
        if 'LN_IC50' not in self.final_df.columns:
            raise ValueError("目標變數 'LN_IC50' 不存在於 final_df 中。")
        y = self.final_df['LN_IC50']

        # 拆分數值特徵
        self.X_train_numeric, self.X_test_numeric, self.y_train, self.y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)

        # 拆分文本特徵 (input_ids 和 attention_mask)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        self.X_train_text = {
            "input_ids": input_ids[:len(self.X_train_numeric)],
            "attention_mask": attention_mask[:len(self.X_train_numeric)],
        }
        self.X_test_text = {
            "input_ids": input_ids[len(self.X_train_numeric):],
            "attention_mask": attention_mask[len(self.X_train_numeric):],
        }

        print("數值特徵與文本特徵分配完成")


    def get_data(self):
        # 數值特徵轉換為 tensor
        X_train_numeric_tensor = torch.tensor(self.X_train_numeric.values).float()
        X_test_numeric_tensor = torch.tensor(self.X_test_numeric.values).float()
        y_train_tensor = torch.tensor(self.y_train.values).float()
        y_test_tensor = torch.tensor(self.y_test.values).float()
        
        # 文本特徵保持 tokenized 格式
        X_train_text = self.X_train_text
        X_test_text = self.X_test_text

        return X_train_numeric_tensor, X_train_text, y_train_tensor, X_test_numeric_tensor, X_test_text, y_test_tensor
