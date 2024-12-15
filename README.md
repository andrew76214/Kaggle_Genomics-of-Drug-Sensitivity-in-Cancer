# Kaggle_Genomics-of-Drug-Sensitivity-in-Cancer
Dataset url : https://www.kaggle.com/datasets/samiraalipour/genomics-of-drug-sensitivity-in-cancer-gdsc/data

## Dataset EDA
### GDSC_DATASET.csv
Origin: 242035 rows x 19 columns
#### Numerical Columns Analysis
* **LN_IC50**: The primary target variable.
* AUC
* Z_SCORE

![image](https://github.com/andrew76214/Kaggle_Genomics-of-Drug-Sensitivity-in-Cancer/blob/main/IMG/GDSC_col_boxplot.png)

This is the box plot of the 3 numerical columns. We could see that there are some outliers, so we removed the data outside the 3 times of IQR.
#### Categorical Columns Analysis
* Microsatellite instability Status (MSI)
* Screen Medium
* Growth Properties
* CNA
* Gene Expression
* Methylation
We mapped these columns form text categories to numbers.
#### Heatmap
![image](https://github.com/andrew76214/Kaggle_Genomics-of-Drug-Sensitivity-in-Cancer/blob/main/IMG/Heatmap.png)

## Evaluation
We use MAE, MSE, RMSE

## Experimental Record
### Multi-Layer Preceptron, MLP
12/03
Best Parameters: {'lr': 0.0001, 'module__hidden_dims': [256, 128, 64], 'optimizer__weight_decay': 0.0001}
Test RMSE: 2.7993
Test MAE: 2.1281
Test MSE: 7.8361

### Deep Learning Model
12/03