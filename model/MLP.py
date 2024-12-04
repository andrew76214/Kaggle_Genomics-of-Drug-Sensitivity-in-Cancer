import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetRegressor
from sklearn.model_selection import GridSearchCV

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