import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

# Wrapper for HistGradientBoosting Regressor
class HistGradientBoosting_Tuner:
    def __init__(self):
        # Instantiate a HistGradientBoosting Regressor
        self.model = HistGradientBoostingRegressor(random_state=42)
        
    def tune_hyperparameters(self, X, y):
        # Define hyperparameter grid
        param_grid = {
            'learning_rate': [0.01, 0.1, 0.2],
            'max_iter': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'min_samples_leaf': [10, 20, 30]
        }
        
        # Use GridSearchCV to find the best hyperparameters
        gs = GridSearchCV(self.model, param_grid, refit=True, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        gs.fit(X, y)
        
        # Print best parameters and score
        print("Best Parameters:", gs.best_params_)
        print("Best Score:", gs.best_score_)
        
        # Return the best model
        return gs.best_estimator_