import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
from sklearn.model_selection import GridSearchCV

# Wrapper for Grid Search
class LGBM_Tuner:
    def __init__(self):
        # Instantiate the LightGBM Regressor model
        self.model = lgb.LGBMRegressor()
        
    def tune_hyperparameters(self, X, y):
        # Define hyperparameter grid
        param_grid = {
            'num_leaves': [31, 50, 70],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200, 500],
            'max_depth': [-1, 10, 20]
        }
        
        # Use GridSearchCV to find the best hyperparameters
        gs = GridSearchCV(self.model, param_grid, refit=True, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        gs.fit(X, y)
        
        # Print best parameters and score
        print("Best Parameters:", gs.best_params_)
        print("Best Score:", gs.best_score_)
        
        # Return the best model
        return gs.best_estimator_
