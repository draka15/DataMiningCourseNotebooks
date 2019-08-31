import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ChronicKidneyDiseaseFull.csv")

nulls_per_column = df.isnull().sum()
nulls_per_column

df_delete_rows = df.dropna(axis=0)
df_delete_rows.shape

df_delete_columns = df.dropna(axis=1)
df_delete_columns.shape

categorical_variables_mask = df.dtypes==object
categorical_variables = df.columns[categorical_variables_mask]
numerical_variables = df.columns[~categorical_variables_mask]



from sklearn_pandas import CategoricalImputer
from sklearn.preprocessing import Imputer

numerical_imputer = Imputer(missing_values="NaN",strategy="median", copy=True)
numerical_imputer.fit(df[numerical_variables])
df_numerical_imputed = numerical_imputer.transform(df[numerical_variables])

categorical_imputer = CategoricalImputer(missing_values="NaN")
categorical_imputer.fit(df[categorical_variables])