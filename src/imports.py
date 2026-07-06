# ruff: noqa: F401
# pyright: ignore[reportMissingImports]
# type: ignore

import os

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import tensorflow as tf
from category_encoders import CountEncoder
from optuna.pruners import MedianPruner
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier as skMLP
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from tensorflow import keras

from data_pipeline import prepare_data
from data_processing import (
    add_datetime_features,
    clean_model_from_sub_and_trim,
    preprocess_split,
    train_val_test_data_split,
)
from MLP import MLP
