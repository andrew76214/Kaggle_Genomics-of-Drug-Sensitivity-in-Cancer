import warnings
warnings.filterwarnings('ignore')

from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV

# Wrapper for Grid Search
class CatBoost_Tuner:
    def __init__(self):
        # Instantiate the CatBoost Regressor model
        self.model = CatBoostRegressor(verbose=0)
        
    def tune_hyperparameters(self, X, y):
        # Define hyperparameter grid
        param_grid = {
            'iterations': [100, 500, 1000],
            'learning_rate': [0.01, 0.05, 0.1],
            'depth': [4, 6, 10],
            'l2_leaf_reg': [1, 3, 5, 7]
        }
        
        # Use GridSearchCV to find the best hyperparameters
        gs = GridSearchCV(self.model, param_grid, refit=True, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        gs.fit(X, y)
        
        # Print best parameters and score
        print("Best Parameters:", gs.best_params_)
        print("Best Score:", gs.best_score_)
        
        # Return the best model
        return gs.best_estimator_
