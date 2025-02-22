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

# toml_path = '/openbayes/home/NEW/rebuild_adrd/data/toml_files/default_conf_new.toml'
toml_path = '/openbayes/home/NEW/rebuild_adrd/data/adni_dataset/adni.toml'
features_and_labels = toml.load(toml_path)
# ic.enable()
# ic(features_and_labels)

feature_column_name = [key for key in features_and_labels['feature'].keys()]

# train_file_path = '/openbayes/home/NEW/rebuild_adrd/data/nacc_train.csv'
# test_file_path = '/openbayes/home/NEW/rebuild_adrd/data/nacc_test.csv'
train_file_path = '/openbayes/home/NEW/rebuild_adrd/data/adni_quchong.csv'
# test_file_path = '/openbayes/home/NEW/rebuild_adrd/data/adni_quchong.csv'

label_column_name = [key for key in features_and_labels['label'].keys()]

train_data = pd.read_csv(train_file_path)
# test_data = pd.read_csv(test_file_path)
# ic(train_data['PTRACCAT'].unique())
# raise()
# randomly split train:test = 8:2
train_data, test_data = train_test_split(train_data, test_size=0.09)
# test_data = train_data[round(train_data.shape[0] * 0.91):]
# train_data = train_data[:round(train_data.shape[0] * 0.91)]

# filtered feature column that not exist in train_data
feature_column_name = [column for column in feature_column_name if column in train_data.columns]

ic(feature_column_name)
feature_column_name.remove('GENOTYPE')
feature_column_name.remove('PTRACCAT')
X_train = train_data[feature_column_name+label_column_name]
correlation_matrix = X_train.corr()

not_none = 0
for col in feature_column_name:
    if train_data[col].isnull().sum() != len(train_data[col]):
        not_none += 1
ic(not_none)
# raise()
ic(len(feature_column_name))
# ic(correlation_matrix['AD']['MCI'])
# ic(correlation_matrix['AD']['NC'])
correlation_with_ad = correlation_matrix['AD'].dropna().sort_values(ascending=False)
ic(correlation_with_ad)
# raise()
for current_name in label_column_name:

    # select X from column name in feature_column_name
    
    X_train = train_data[feature_column_name]
    X_test = test_data[feature_column_name]

    # select y from column name in label_column_name
    y_train = train_data[current_name]
    y_test = test_data[current_name]

    # print("X_train shape:", X_train.shape)
    # print("y_train shape:", y_train.shape)
    # print("X_test shape:", X_test.shape)
    # print("y_test shape:", y_test.shape)

    row_number_array = [3, 10, 30, 100, 300, 1000, 3000]

    # if X_train.shape[0] >= 10000:
    #     # sample 10000 rows id in range(0, X_train.shape[0])
    #     sample_index = np.random.choice(X_train.shape[0], 10000, replace=False)
    #     X_train = X_train.iloc[sample_index]
    #     y_train = y_train.iloc[sample_index]

    for row_number in row_number_array:
        torch.cuda.empty_cache()
        classifier = TabPFNClassifier(random_state=42)
        while(True):
            sample_index = np.random.choice(X_train.shape[0], row_number, replace=False)
            X_train_subset = X_train.iloc[sample_index].copy()
            y_train_subset = y_train.iloc[sample_index].copy()

            # clean GPU cache
            assert(X_train_subset.shape[0] != 0)
            # check if X_train_subset contains NaN
            # ic(X_train_subset['MH16BSMOK'].isnull().sum())
            # if y_train_subset not all null break
            if y_train_subset.isnull().sum() != row_number:
                break

        to_drop = []
        for col in X_train_subset.columns:
            # ic(col)
            # ic(col, X_train_subset[col].dtype)
            if X_train_subset[col].dtype != np.float64:
                # ic(col, X_train_subset[col])``
                to_drop.append(col)
            elif X_train_subset[col].isnull().sum() == X_train_subset.shape[0]:
                to_drop.append(col)
        # raise()
        # drop all columns in to_drop
        X_train_subset.drop(columns=to_drop, inplace=True)
        X_new_test = X_test.drop(columns=to_drop, inplace=False)
        # assert(X_train_subset.isnull().sum().sum() == 0)

        # ic(row_number)
        classifier.fit(X_train_subset, y_train_subset)
        y_pred = classifier.predict_proba(X_new_test)
        # print(y_pred.shape)
        score = roc_auc_score(y_test, y_pred[:, 1] if y_pred.shape[1] > 1 else y_pred[:, 0])
        # print(f"TabPFN [{current_name}] ROC AUC: {score:.4f}")
        print(f"TabPFN [{current_name}] ROC AUC: {score:.4f} with {row_number} rows")

def clean_test_data(test_data):
    # clean test data
    to_drop = []
    for col in test_data.columns:
        if test_data[col].dtype != np.float64:
            to_drop.append(col)
        elif test_data[col].isnull().sum() == test_data.shape[0]:
            to_drop.append(col)
    test_data.drop(columns=to_drop, inplace=True)
    return test_data

def sample_rows(X_train, y_train, row_number):
    while(True):
        sample_index = np.random.choice(X_train.shape[0], row_number, replace=False)
        X_train_subset = X_train.iloc[sample_index].copy()
        y_train_subset = y_train.iloc[sample_index].copy()
        if y_train_subset.isnull().sum() != row_number:
            break
    to_drop = []
    for col in X_train_subset.columns:
        if X_train_subset[col].dtype != np.float64:
            to_drop.append(col)
        elif X_train_subset[col].isnull().sum() == X_train_subset.shape[0]:
            to_drop.append(col)
    X_train_subset.drop(columns=to_drop, inplace=True)
    return X_train_subset, y_train_subset
