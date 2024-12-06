import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV

# Wrapper for ElasticNet Regressor
class ElasticNet_Tuner:
    def __init__(self):
        # Instantiate an ElasticNet Regressor
        self.model = ElasticNet(random_state=42)
        
    def tune_hyperparameters(self, X, y):
        # Define hyperparameter grid
        param_grid = {
            'alpha': [0.1, 1.0, 10.0],
            'l1_ratio': [0.1, 0.5, 0.9],
            'max_iter': [1000, 5000, 10000],
        }
        
        # Use GridSearchCV to find the best hyperparameters
        gs = GridSearchCV(self.model, param_grid, refit=True, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        gs.fit(X, y)
        
        # Print best parameters and score
        print("Best Parameters:", gs.best_params_)
        print("Best Score:", gs.best_score_)
        
        # Return the best model
        return gs.best_estimator_