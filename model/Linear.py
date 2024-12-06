import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

# Wrapper for Linear Regression
class LinearRegression_Tuner:
    def __init__(self):
        # Instantiate a Linear Regression model
        self.model = LinearRegression()
        
    def tune_hyperparameters(self, X, y):
        # Define hyperparameter grid
        param_grid = {
            'fit_intercept': [True, False],
            'normalize': [True, False]
        }
        
        # Use GridSearchCV to find the best hyperparameters
        gs = GridSearchCV(self.model, param_grid, refit=True, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        gs.fit(X, y)
        
        # Print best parameters and score
        print("Best Parameters:", gs.best_params_)
        print("Best Score:", gs.best_score_)
        
        # Return the best model
        return gs.best_estimator_