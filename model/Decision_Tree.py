import warnings
warnings.filterwarnings('ignore')

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

# Wrapper for Grid Search
class DecisionTree_Tuner:
    def __init__(self):
        # Instantiate the Decision Tree Regressor model
        self.model = DecisionTreeRegressor()
        
    def tune_hyperparameters(self, X, y):
        # Define hyperparameter grid
        param_grid = {
            'criterion': ['mse', 'friedman_mse', 'mae', 'poisson'],
            'max_depth': [None, 10, 20, 30, 40],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Use GridSearchCV to find the best hyperparameters
        gs = GridSearchCV(self.model, param_grid, refit=True, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        gs.fit(X, y)
        
        # Print best parameters and score
        print("Best Parameters:", gs.best_params_)
        print("Best Score:", gs.best_score_)
        
        # Return the best model
        return gs.best_estimator_