import warnings 
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetRegressor
from sklearn.model_selection import GridSearchCV 


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
            batch_size=512,
            iterator_train__shuffle=True,
            train_split=None,
            verbose=1,
            device='cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
        )

    def tune_hyperparameters(self, X, y):
        # Define hyperparameter grid
        param_grid = {
            'lr': [0.01, 0.001],
            'optimizer__weight_decay': [0.0001, 0.001],
            'module__cnn_filters': [32, 64, 128],
            'module__transformer_dim': [64, 128],
            'module__nhead': [2],
            'module__num_layers': [1, 2],
        }

        # Use GridSearchCV to find the best hyperparameters
        gs = GridSearchCV(self.model, param_grid, refit=True, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        gs.fit(X, y)

        # Print best parameters and score
        print("Best Parameters:", gs.best_params_)
        print("Best Score:", gs.best_score_)

        # Return the best model
        return gs.best_estimator_