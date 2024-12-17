# Kaggle_Genomics-of-Drug-Sensitivity-in-Cancer
Dataset url : https://www.kaggle.com/datasets/samiraalipour/genomics-of-drug-sensitivity-in-cancer-gdsc/data
## Introduction
* This dataset is from the GDSC project, which is a collaboration between the UK and the USA.
* The project involves large-scale screening of human cancer cell lines with a wide range of anti-cancer drugs. In the experiments, cell viability was checked using the CellTiter-Glo test after 72 hours of drug treatment.
* The GDSC dataset provides a comprehensive resource for studying the response of various cancer cell lines to a wide range of anti-cancer drugs.
* The target variable in this dataset is LN_IC50
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
## Training Pipeline
![image](https://github.com/andrew76214/Kaggle_Genomics-of-Drug-Sensitivity-in-Cancer/blob/main/IMG/training_pipeline.png)
## Data Preprocess
### Step 1: dropna operation
Apply the dropna operation to remove missing values and ensure data completeness.
### Step 2: first merge
Merge GDSC2_DATASET.csv with Cell_Lines_Details.xlsx to create a new dataset: merged_df.
### Step 3: second merge
Further merge merged_df with Compounds_annotation.csv to produce the final dataset: final_df.
## Evaluation
We use MAE, MSE, RMSE

## Folder Architecture
```
Kaggle_Genomics-of-Drug-Sensitivity-in-Cancer/
│
├── README.md
├── DatasetEDA.py
├── LICENSE
├── __pycache__
│
├── BERT/
│   ├── __pycache__
│   ├── main.ipynb
│   ├── data_loader_BERT.py
│   ├── model.py
│   ├── config.py
│   └── utils.py
│
├── DL/
│   ├── main_DL.py
│   ├── dataloader.py
│   ├── MLP.py
│   └── CNNTransformer.py
│
├── IMG/
│   ├── GDSC_col_boxplot.png
│   ├── Heatmap.png
│   └── training_pipeline.png
│
└── ML/
    ├── main_ML.py
    ├── dataloader.py
    ├── ML_model.py
    └── tuning_log.md
```

## Experimental Record
### Multi-Layer Preceptron, MLP
12/03
Best Parameters: {'lr': 0.0001, 'module__hidden_dims': [256, 128, 64], 'optimizer__weight_decay': 0.0001}
Test RMSE: 2.7993
Test MAE: 2.1281
Test MSE: 7.8361

### Deep Learning Model
12/03

---
### Contributors

<a href="https://github.com/andrew76214/Kaggle_Genomics-of-Drug-Sensitivity-in-Cancer/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=andrew76214/Kaggle_Genomics-of-Drug-Sensitivity-in-Cancer" />
</a>

Made with [contrib.rocks](https://contrib.rocks).