import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# Wrapper for Grid Search
class XGBRegressor_Tuner:
    def __init__(self):
        # Instantiate the XGBoost Regressor model
        self.model = xgb.XGBRegressor(objective='reg:squarederror')
        
    def tune_hyperparameters(self, X, y):
        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }
        
        # Use GridSearchCV to find the best hyperparameters
        gs = GridSearchCV(self.model, param_grid, refit=True, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        gs.fit(X, y)
        
        # Print best parameters and score
        print("Best Parameters:", gs.best_params_)
        print("Best Score:", gs.best_score_)
        
        # Return the best model
        return gs.best_estimator_