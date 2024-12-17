import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class DatasetEDA:
    def __init__(self, path):
        self.df = pd.read_csv(path)
        self.df = self.df.rename(columns={'Microsatellite instability Status (MSI)':'MSI', 'GDSC Tissue descriptor 1':'Descriptor 1', 'GDSC Tissue descriptor 2':'Descriptor 2', 'Cancer Type (matching TCGA label)':'Cancer Type'})
        self.num_col = ['LN_IC50', 'AUC', 'Z_SCORE']
        self.categoty_col = ['MSI', 'Screen Medium', 'Growth Properties']
        self.binary_col = ['CNA', 'Gene Expression', 'Methylation']
        self.preprocess()

    def print_info(self):
        print(self.df.head(), "\n")
        print(self.df.info(), "\n")
        print(self.df.describe(), "\n")

    def preprocess(self):
        self.df = self.df.drop_duplicates()
        self.df = self.df.dropna()

    def box_plot(self):
        plt.figure()
        plt.subplot(3,1,1)
        sns.boxplot(x=self.df['LN_IC50'])
        plt.title('LN_IC50')
        plt.gca().set(xlabel=None, ylabel=None)
        plt.subplot(3,1,2)
        sns.boxplot(x=self.df['AUC'])
        plt.title('AUC')
        plt.gca().set(xlabel=None, ylabel=None)
        plt.subplot(3,1,3)
        sns.boxplot(x=self.df['Z_SCORE'])
        plt.title('Z_SCORE')
        plt.gca().set(xlabel=None, ylabel=None)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.show()

    def remove_outliers(self, times=3):
        Q1 = self.df[self.num_col].quantile(0.25)
        Q3 = self.df[self.num_col].quantile(0.75)
        IQR = Q3 - Q1
        self.df = self.df[~((self.df[self.num_col] < (Q1 - times * IQR)) |(self.df[self.num_col] > (Q3 + times * IQR))).any(axis=1)]
        print(self.df.count(), "\n")

    def print_categorical_col(self):
        print("TCGA_DESC:\n", self.df['TCGA_DESC'].value_counts(), "\n")
        print("desc1:\n", self.df['Descriptor 1'].value_counts(), "\n")
        print("desc2:\n", self.df['Descriptor 2'].value_counts(), "\n")
        print("Cancer Type:\n", self.df['Cancer Type'].value_counts(), "\n")
        print("MSI:\n", self.df['MSI'].value_counts(), "\n")
        print("Screen:\n", self.df['Screen Medium'].value_counts(), "\n")
        print("Growth:\n", self.df['Growth Properties'].value_counts(), "\n")
        print("CNA:\n", self.df['CNA'].value_counts(), "\n")
        print("Gene Expression:\n", self.df['Gene Expression'].value_counts(), "\n")
        print("Methylation:\n", self.df['Methylation'].value_counts(), "\n")
        print("Target:\n", self.df['TARGET'].value_counts(), "\n")
        print("Target pathway:\n", self.df['TARGET_PATHWAY'].value_counts(), "\n")

    def map_obj2num(self):
        self.df['MSI'] = self.df['MSI'].map({'MSS/MSI-L':0, 'MSI-H':1})
        self.df['Screen Medium'] = self.df['Screen Medium'].map({'R':0, 'D/F12':1})
        self.df['Growth Properties'] = self.df['Growth Properties'].map({'Adherent':0, 'Suspension':1, 'Semi-Adherent':2})
        for col in self.binary_col:
            self.df[col] = self.df[col].map({'N': 0, 'Y': 1})

    def heatmap(self):
        plt.figure()
        df = self.df.drop(columns=['COSMIC_ID', 'DRUG_ID'])
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='Reds')
        plt.title("Heatmap of GDSC columns")
        plt.subplots_adjust(left=0.3, bottom=0.3)
        plt.show()

if __name__ == "__main__":
    path = "./GDSC_DATASET.csv"
    EDA = DatasetEDA(path)
    EDA.remove_outliers()
    EDA.map_obj2num()
    EDA.heatmap()