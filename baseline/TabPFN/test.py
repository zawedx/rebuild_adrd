import os

# Setup Imports
import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from IPython.display import display, Markdown, Latex

# Baseline Imports
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

import torch

from tabpfn import TabPFNClassifier, TabPFNRegressor

if not torch.cuda.is_available():
    raise SystemError('GPU device not found. For fast training, please enable GPU. See section above for instructions.')
import warnings
# import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)

# from tabpfn_extensions.hpo import (
#     TunedTabPFNRegressor,
#     TunedTabPFNClassifier,
# )

import pandas as pd
import toml
from icecream import ic

def clean_test_data(test_data):
    # clean test data
    to_drop = []
    for col in test_data.columns:
        if test_data[col].dtype != np.float64 and test_data[col].dtype != np.int64:
            to_drop.append(col)
            if col == 'NC':
                ic(test_data[col].dtype)
        elif test_data[col].isnull().sum() == test_data.shape[0]:
            to_drop.append(col)
            if col == 'NC':
                ic([test_data[col].isnull().sum(), test_data.shape[0]])
    test_data.drop(columns=to_drop, inplace=True)
    return test_data

# toml_path = '/openbayes/home/NEW/rebuild_adrd/data/toml_files/default_conf_new.toml'
toml_path = '/openbayes/home/NEW/rebuild_adrd/data/adni_dataset/adni.toml'
features_and_labels = toml.load(toml_path)

feature_column_name = [key for key in features_and_labels['feature'].keys()]

# train_file_path = '/openbayes/home/NEW/rebuild_adrd/data/nacc_train.csv'
# test_file_path = '/openbayes/home/NEW/rebuild_adrd/data/nacc_test.csv'
train_file_path = '/openbayes/home/NEW/rebuild_adrd/data/adni_quchong.csv'
# test_file_path = '/openbayes/home/NEW/rebuild_adrd/data/adni_quchong.csv'

label_column_name = [key for key in features_and_labels['label'].keys()]

train_data = pd.read_csv(train_file_path)
# test_data = pd.read_csv(test_file_path)

# filtered feature column that not exist in train_data
feature_column_name = [column for column in feature_column_name if column in train_data.columns]

feature_column_name.remove('GENOTYPE')
feature_column_name.remove('PTRACCAT')
train_data = train_data[feature_column_name+label_column_name]

ic('NC' in train_data.columns)
ic(train_data['NC'])
train_data = clean_test_data(train_data)
ic('NC' in train_data.columns)

train_data, test_data = train_test_split(train_data, test_size=0.09)
ic('NC' in train_data.columns)
ic('NC' in test_data.columns)

# record the result for different label and different row number, repeat T=10 times
T = 10
result = {}

for current_name in label_column_name:
    # select X from column name in feature_column_name
    X_train = train_data[feature_column_name]
    X_test = test_data[feature_column_name]

    # select y from column name in label_column_name
    
    y_train = train_data[current_name]
    y_test = test_data[current_name]

    row_number_array = [3, 10, 30, 100, 300, 1000, 3000]

    for row_number in row_number_array:
        for i in range(T):
            torch.cuda.empty_cache()
            classifier = TabPFNClassifier(random_state=42)
            while(True):
                sample_index = np.random.choice(X_train.shape[0], row_number, replace=False)
                X_train_subset = X_train.iloc[sample_index].copy()
                y_train_subset = y_train.iloc[sample_index].copy()
                assert(X_train_subset.shape[0] != 0)
                if y_train_subset.isnull().sum() != row_number:
                    break

            classifier.fit(X_train_subset, y_train_subset)
            y_pred = classifier.predict_proba(X_test)
            score = roc_auc_score(y_test, y_pred[:, 1] if y_pred.shape[1] > 1 else y_pred[:, 0])
            print(f"TabPFN [{current_name}] ROC AUC: {score:.4f} with {row_number} rows")
            if current_name not in result:
                result[current_name] = {}
            if row_number not in result[current_name]:
                result[current_name][row_number] = []
            result[current_name][row_number].append(score)

# ic(result)
# for each label, plot boxplot for different row number
for current_name in label_column_name:
    plt.figure(figsize=(10, 6))
    plt.title(f"{current_name} ROC AUC")
    plt.boxplot([result[current_name][row_number] for row_number in row_number_array], tick_labels=row_number_array)
    plt.xlabel("Number of Rows")
    plt.ylabel("ROC AUC")
    plt.show()
    # save the plot
    plt.savefig(f"TabPFN_{current_name}_ROC_AUC.png")