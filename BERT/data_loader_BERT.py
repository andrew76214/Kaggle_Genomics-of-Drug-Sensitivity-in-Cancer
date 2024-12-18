import pandas as pd
import torch
from transformers import BertTokenizer, RobertaTokenizer, DebertaV2Tokenizer
from sklearn.model_selection import train_test_split
from config import BERT_MODEL_NAME


class DataLoader4BERT:
    def __init__(self, gdsc_path, compounds_path, gdsc2_path, cell_lines_path):
        self.gdsc_path = gdsc_path
        self.compounds_path = compounds_path
        self.gdsc2_path = gdsc2_path
        self.cell_lines_path = cell_lines_path
        self.bert_model_name = BERT_MODEL_NAME
        self.tokenizer = self._initialize_tokenizer(self.bert_model_name)
        
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
        
    def _initialize_tokenizer(self, model_name):
        if 'roberta-base' in model_name:
            return RobertaTokenizer.from_pretrained(model_name)
        elif 'deberta' in model_name:
            return DebertaV2Tokenizer.from_pretrained(model_name)
        else:  # Default to BertTokenizer
            return BertTokenizer.from_pretrained(model_name)

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