************************ <class 'ML_model.DecisionTree_Tuner'> ************************
Best Parameters: {'criterion': 'friedman_mse', 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 10}
Best Score: -0.17322727011780784
Test RMSE: 0.3585
Test MAE: 0.1707
Test MSE: 0.1285
************************ <class 'ML_model.CatBoost_Tuner'> ************************
Error with model <class 'ML_model.CatBoost_Tuner'>: 
All the 324 fits failed.
It is very likely that your model is misconfigured.
You can try to debug the error by setting error_score='raise'.

Below are more details about the failures:
--------------------------------------------------------------------------------
324 fits failed with the following error:
Traceback (most recent call last):
  File "/home/andrew-root/miniconda3/envs/pytorch/lib/python3.12/site-packages/sklearn/model_selection/_validation.py", line 888, in _fit_and_score
    estimator.fit(X_train, y_train, **fit_params)
  File "/home/andrew-root/miniconda3/envs/pytorch/lib/python3.12/site-packages/catboost/core.py", line 5873, in fit
    return self._fit(X, y, cat_features, text_features, embedding_features, None, graph, sample_weight, None, None, None, None, baseline,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/andrew-root/miniconda3/envs/pytorch/lib/python3.12/site-packages/catboost/core.py", line 2395, in _fit
    train_params = self._prepare_train_params(
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/andrew-root/miniconda3/envs/pytorch/lib/python3.12/site-packages/catboost/core.py", line 2275, in _prepare_train_params
    train_pool = _build_train_pool(X, y, cat_features, text_features, embedding_features, pairs, graph,
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/andrew-root/miniconda3/envs/pytorch/lib/python3.12/site-packages/catboost/core.py", line 1513, in _build_train_pool
    train_pool = Pool(X, y, cat_features=cat_features, text_features=text_features, embedding_features=embedding_features, pairs=pairs, graph=graph, weight=sample_weight, group_id=group_id,
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/andrew-root/miniconda3/envs/pytorch/lib/python3.12/site-packages/catboost/core.py", line 768, in __init__
    self._check_data_type(data)
  File "/home/andrew-root/miniconda3/envs/pytorch/lib/python3.12/site-packages/catboost/core.py", line 925, in _check_data_type
    raise CatBoostError(
_catboost.CatBoostError: Invalid data type=<class 'torch.Tensor'>: data must be list(), np.ndarray(), DataFrame(), Series(), FeaturesData  scipy.sparse matrix or filename str() or pathlib.Path().

************************ <class 'ML_model.ElasticNet_Tuner'> ************************
Best Parameters: {'alpha': 0.1, 'l1_ratio': 0.9, 'max_iter': 1000}
Best Score: -4.111563349305744
Test RMSE: 2.0207
Test MAE: 1.4311
Test MSE: 4.0831
************************ <class 'ML_model.HistGradientBoosting_Tuner'> ************************
Best Parameters: {'learning_rate': 0.2, 'max_depth': 7, 'max_iter': 300, 'min_samples_leaf': 10}
Best Score: -0.08140303563457703
Test RMSE: 0.2793
Test MAE: 0.2028
Test MSE: 0.0780
************************ <class 'ML_model.KNN_Tuner'> ************************
Best Parameters: {'algorithm': 'ball_tree', 'n_neighbors': 3, 'p': 2, 'weights': 'distance'}
Best Score: -0.23797186810388235
Test RMSE: 0.4049
Test MAE: 0.1828
Test MSE: 0.1639
************************ <class 'ML_model.Lasso_Tuner'> ************************
Best Parameters: {'alpha': 0.01, 'max_iter': 1000}
Best Score: -1.617123431897694
Test RMSE: 1.2684
Test MAE: 0.9700
Test MSE: 1.6088
************************ <class 'ML_model.LGBM_Tuner'> ************************
[LightGBM] [Info] Total Bins 1728
[LightGBM] [Info] Number of data points in the train set: 171060, number of used features: 611
[LightGBM] [Info] Start training from score 2.777687
Best Parameters: {'learning_rate': 0.1, 'max_depth': -1, 'n_estimators': 500, 'num_leaves': 70}
Best Score: -0.02643431104263098
Test RMSE: 0.1508
Test MAE: 0.0998
Test MSE: 0.0228
************************ <class 'ML_model.LinearRegression_Tuner'> ************************
Best Parameters: {'copy_X': True, 'fit_intercept': False}
Best Score: -0.09719690432151158
Test RMSE: 0.3082
Test MAE: 0.2042
Test MSE: 0.0950
************************ <class 'ML_model.RF_Tuner'> ************************
Best Parameters: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}
Best Score: -0.10655945577142985
Test RMSE: 0.2859
Test MAE: 0.1482
Test MSE: 0.0818
************************ <class 'ML_model.Ridge_Tuner'> ************************
Best Parameters: {'alpha': 0.01, 'solver': 'auto'}
Best Score: -0.09719565820429625
Test RMSE: 0.3080
Test MAE: 0.2040
Test MSE: 0.0949
************************ <class 'ML_model.XGBRegressor_Tuner'> ************************
Best Parameters: {'colsample_bytree': 1.0, 'learning_rate': 0.2, 'max_depth': 10, 'n_estimators': 300, 'subsample': 0.6}
Best Score: -0.04561541477839152
