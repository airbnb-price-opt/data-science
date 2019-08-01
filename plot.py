import pandas as pd 
import numpy as np
#import shap
import os
import dill as pickle

def zip_list (zipcode):

    df_plot = pd.read_csv('data-clean.csv')

    bins = [0,50,100,150,200,300,400,500,750,1000,50000]

    df_plot = df_plot.loc[df_plot['zipcode'] ==zipcode]

    digitized = np.digitize(df_plot['total_price'], bins)
    bin_counts = [df_plot['total_price'][digitized == i].count() for i in range(1, len(bins))]

    return bin_counts, bins

"""
def shap_list():
  with open('./model_v1.pk', 'rb') as f:
    model = pickle.load(f)
  # load JS visualization code to notebook
  shap.initjs()
  # explain the model's predictions using SHAP values
  # (same syntax works for LightGBM, CatBoost, and scikit-learn models)
  explainer = shap.TreeExplainer(model)
  shap_values = explainer.shap_values(X)
  return shap_values
"""
