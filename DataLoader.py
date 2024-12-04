import torch
import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader:
        # Usage example:
    # dataloader = DataLoader('./genomics-of-drug-sensitivity-in-cancer-gdsc/GDSC_DATASET.csv',
    #                         './genomics-of-drug-sensitivity-in-cancer-gdsc/Compounds-annotation.csv',
    #                         './genomics-of-drug-sensitivity-in-cancer-gdsc/GDSC2-dataset.csv',
    #                         './genomics-of-drug-sensitivity-in-cancer-gdsc/Cell_Lines_Details.xlsx')
    # dataloader.load_data()
    # dataloader.preprocess_data()
    # dataloader.define_features_and_target()
    # X_train, X_test, y_train, y_test = dataloader.get_data()
    
    def __init__(self, gdsc_path, compounds_path, gdsc2_path, cell_lines_path):
        self.gdsc_path = gdsc_path
        self.compounds_path = compounds_path
        self.gdsc2_path = gdsc2_path
        self.cell_lines_path = cell_lines_path
        self.gdsc_dataset = None
        self.compounds_annotation = None
        self.gdsc2_dataset = None
        self.cell_lines_details = None
        self.final_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        # Load the datasets
        self.gdsc_dataset = pd.read_csv(self.gdsc_path)
        self.compounds_annotation = pd.read_csv(self.compounds_path)
        self.gdsc2_dataset = pd.read_csv(self.gdsc2_path)
        self.cell_lines_details = pd.read_excel(self.cell_lines_path)

    def preprocess_data(self):
        # Drop rows with missing values for simplicity
        self.gdsc_dataset = self.gdsc_dataset.dropna()
        self.compounds_annotation = self.compounds_annotation.dropna()
        self.gdsc2_dataset = self.gdsc2_dataset.dropna()
        self.cell_lines_details = self.cell_lines_details.dropna()

        # Merge GDSC2 dataset with Cell Lines Details on COSMIC_ID
        merged_df = pd.merge(self.gdsc2_dataset, self.cell_lines_details, left_on='COSMIC_ID', right_on='COSMIC identifier', how='left')

        # Merge the resulting dataframe with Compounds Annotation on DRUG_ID
        self.final_df = pd.merge(merged_df, self.compounds_annotation, on='DRUG_ID', how='left')

    def define_features_and_target(self):
        # Define features and target variable
        X = self.final_df[['DRUG_ID', 'AUC', 'Z_SCORE', 
                           'Whole Exome Sequencing (WES)',
                           'Copy Number Alterations (CNA)',
                           'Gene Expression',
                           'Methylation',
                           'DRUG_ID',
                           'GDSC\nTissue descriptor 1',
                           'GDSC\nTissue\ndescriptor 2',
                           'Cancer Type\n(matching TCGA label)', 
                           'Microsatellite \ninstability Status (MSI)',
                           'Growth Properties']]

        y = self.final_df['LN_IC50']

        # Hot one encoding variable that need to be hot-one encoded
        X = pd.get_dummies(X, columns=[
            'Whole Exome Sequencing (WES)',
            'Copy Number Alterations (CNA)',
            'Gene Expression',
            'Methylation',
            'DRUG_ID',
            'GDSC\nTissue descriptor 1',
            'GDSC\nTissue\ndescriptor 2',
            'Cancer Type\n(matching TCGA label)', 
            'Microsatellite \ninstability Status (MSI)',
            'Growth Properties', 
        ])

        X = X.astype(float)

        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def Convert_to_tensors(self):
        
        input_dim = self.X_train.shape[1]
        
        X_train_tensor = torch.tensor(self.X_train.values).float()
        y_train_tensor = torch.tensor(self.y_train.values).float()
        X_test_tensor = torch.tensor(self.X_test.values).float()
        y_test_tensor = torch.tensor(self.y_test.values).float()

        return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, input_dim