import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from skorch import NeuralNetRegressor
from sklearn.model_selection import GridSearchCV

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


# Model Implementation
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32]):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.fc4 = nn.Linear(hidden_dims[2], 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Wrapper for Grid Search
class MLP_Tuner:
    def __init__(self, input_dim, model_class=MLP):
        self.input_dim = input_dim
        self.model = NeuralNetRegressor(
            model_class,
            module__input_dim=input_dim,
            criterion=nn.MSELoss,
            optimizer=optim.Adam,
            max_epochs=30,
            batch_size=128,
            iterator_train__shuffle=True,
            train_split=None,
            verbose=1,
            device='cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
        )
        
    def tune_hyperparameters(self, X, y):
        # Define hyperparameter grid
        param_grid = {
            'lr': [0.01, 0.001, 0.0001],
            'optimizer__weight_decay': [0, 0.0001, 0.001],
            'module__hidden_dims': [[128, 64, 32], [256, 128, 64], [512, 256, 128], [512, 256, 128, 64]],
        }
        
        # Use GridSearchCV to find the best hyperparameters
        gs = GridSearchCV(self.model, param_grid, refit=True, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        gs.fit(X, y)
        
        # Print best parameters and score
        print("Best Parameters:", gs.best_params_)
        print("Best Score:", gs.best_score_)
        
        # Return the best model
        return gs.best_estimator_
    
  
class CNNTransformer(nn.Module):
    def __init__(self, input_dim, cnn_filters, transformer_dim, nhead, num_layers, seq_len):
        super(CNNTransformer, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, cnn_filters, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(cnn_filters)
        self.conv2 = nn.Conv1d(cnn_filters, cnn_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(cnn_filters)
        self.conv3 = nn.Conv1d(cnn_filters, cnn_filters * 2, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(cnn_filters * 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.input_layer = nn.Linear(cnn_filters * 2, transformer_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=nhead, dim_feedforward=transformer_dim * 4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(transformer_dim, 1)

    '''def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout(self.relu(self.bn3(self.conv3(x))))

        x = x.permute(0, 2, 1)
        x = self.input_layer(x)
        x = self.transformer_encoder(x)

        x_mean = x.mean(dim=1)
        x_output = self.fc(x_mean)

        return x_output'''
    def forward(self, x):
        # Ensure input has the right shape
        if x.dim() == 2:  # Add sequence dimension if missing
            x = x.unsqueeze(1)  # [batch_size, 1, input_dim]

        # Conv1d expects [batch_size, input_channels, seq_len]
        x = x.permute(0, 2, 1)  # [batch_size, input_dim, seq_len]

        # Convolutional layers
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout(self.relu(self.bn3(self.conv3(x))))

        # Permute back for transformer
        x = x.permute(0, 2, 1)  # [batch_size, seq_len, cnn_filters*2]

        # Linear layer to match transformer input dim
        x = self.input_layer(x)  # [batch_size, seq_len, transformer_dim]

        # Transformer expects [seq_len, batch_size, embed_dim]
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, transformer_dim]

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Mean pooling over sequence length
        x_mean = x.mean(dim=0)  # [batch_size, transformer_dim]

        # Final fully connected layer
        x_output = self.fc(x_mean)

        return x_output


  
class CNNTransformer_Tuner:
    def __init__(self, input_dim, cnn_filters=64, transformer_dim=256, nhead=8, num_layers=3, seq_len=1291, model_class=CNNTransformer):
        self.input_dim = input_dim
        self.cnn_filters = cnn_filters
        self.transformer_dim = transformer_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self.seq_len = seq_len

        self.model = NeuralNetRegressor(
            model_class,
            module__input_dim=input_dim,
            module__cnn_filters=cnn_filters,
            module__transformer_dim=transformer_dim,
            module__nhead=nhead,
            module__num_layers=num_layers,
            module__seq_len=seq_len,
            criterion=nn.MSELoss,
            optimizer=optim.Adam,
            max_epochs=30,
            batch_size=256,
            iterator_train__shuffle=True,
            train_split=None,
            verbose=1,
            device='cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
        )

    def tune_hyperparameters(self, X, y):
        # Define hyperparameter grid
        param_grid = {
            'lr': [0.01, 0.001, 0.0001],
            'optimizer__weight_decay': [0, 0.0001, 0.001],
            'module__cnn_filters': [32, 64, 128],
            'module__transformer_dim': [64, 128, 256],
            'module__nhead': [2, 4, 8],
            'module__num_layers': [1, 2, 3],
        }

        # Use GridSearchCV to find the best hyperparameters
        gs = GridSearchCV(self.model, param_grid, refit=True, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        gs.fit(X, y)

        # Print best parameters and score
        print("Best Parameters:", gs.best_params_)
        print("Best Score:", gs.best_score_)

        # Return the best model
        return gs.best_estimator_