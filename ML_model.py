import warnings
warnings.filterwarnings('ignore')

from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb

from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor


# Wrapper for XGBoost
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
        return gs.best_estimator_, gs.best_params_, gs.best_score_

# Wrapper for SVR
class SVR_Tuner:
    def __init__(self):
        # Instantiate the SVR model
        self.model = SVR()
        
    def tune_hyperparameters(self, X, y):
        # Define hyperparameter grid
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto'],
            'epsilon': [0.1, 0.01, 0.001]
        }
        
        # Use GridSearchCV to find the best hyperparameters
        gs = GridSearchCV(self.model, param_grid, refit=True, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        gs.fit(X, y)
        
        # Print best parameters and score
        print("Best Parameters:", gs.best_params_)
        print("Best Score:", gs.best_score_)
        
        # Return the best model
        return gs.best_estimator_, gs.best_params_, gs.best_score_

# Wrapper for Ridge Regressor
class Ridge_Tuner:
    def __init__(self):
        # Instantiate a Ridge Regressor
        self.model = Ridge(random_state=42)
        
    def tune_hyperparameters(self, X, y):
        # Define hyperparameter grid
        param_grid = {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg'],
        }
        
        # Use GridSearchCV to find the best hyperparameters
        gs = GridSearchCV(self.model, param_grid, refit=True, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        gs.fit(X, y)
        
        # Print best parameters and score
        print("Best Parameters:", gs.best_params_)
        print("Best Score:", gs.best_score_)
        
        # Return the best model
        return gs.best_estimator_, gs.best_params_, gs.best_score_

# Wrapper for Random Forest
class RF_Tuner:
    def __init__(self):
        # Instantiate the Random Forest Regressor model
        self.model = RandomForestRegressor()
        
    def tune_hyperparameters(self, X, y):
        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
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
        return gs.best_estimator_, gs.best_params_, gs.best_score_

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
        return gs.best_estimator_, gs.best_params_, gs.best_score_

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
        return gs.best_estimator_, gs.best_params_, gs.best_score_

# Wrapper for Lasso Regressor
class Lasso_Tuner:
    def __init__(self):
        # Instantiate a Lasso Regressor
        self.model = Lasso(random_state=42)
        
    def tune_hyperparameters(self, X, y):
        # Define hyperparameter grid
        param_grid = {
            'alpha': [0.01, 0.1, 1.0, 10.0],
            'max_iter': [1000, 5000, 10000],
        }
        
        # Use GridSearchCV to find the best hyperparameters
        gs = GridSearchCV(self.model, param_grid, refit=True, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        gs.fit(X, y)
        
        # Print best parameters and score
        print("Best Parameters:", gs.best_params_)
        print("Best Score:", gs.best_score_)
        
        # Return the best model
        return gs.best_estimator_, gs.best_params_, gs.best_score_

# Wrapper for KNN
class KNN_Tuner:
    def __init__(self):
        # Instantiate the KNeighborsRegressor model
        self.model = KNeighborsRegressor()
        
    def tune_hyperparameters(self, X, y):
        # Define hyperparameter grid
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'p': [1, 2]
        }
        
        # Use GridSearchCV to find the best hyperparameters
        gs = GridSearchCV(self.model, param_grid, refit=True, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        gs.fit(X, y)
        
        # Print best parameters and score
        print("Best Parameters:", gs.best_params_)
        print("Best Score:", gs.best_score_)
        
        # Return the best model
        return gs.best_estimator_, gs.best_params_, gs.best_score_

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
        return gs.best_estimator_, gs.best_params_, gs.best_score_

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
        return gs.best_estimator_, gs.best_params_, gs.best_score_

# Wrapper for CatBoost
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
        return gs.best_estimator_, gs.best_params_, gs.best_score_

# Wrapper for DecisionTree
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
        return gs.best_estimator_, gs.best_params_, gs.best_score_