import os
import numpy as np
import pandas as pd
import dill as pickle
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import category_encoders as ce
from sklearn.linear_model import RidgeCV
import pathlib

import warnings
warnings.filterwarnings("ignore")


def transform_data(df, num_cols, cat_cols):
    """
    Creates transformed dataframe with:
        -One-hot-encoded categorical features
        -Scaled numerical features
    Creates list with all column names of transformed dataframe
    """
    df_num = df[num_cols]

    if len(cat_cols) == 1:
        df_cat = pd.DataFrame(df, columns=cat_cols)
    else:
        df_cat = df[cat_cols]

    binary_encoder = ce.binary.BinaryEncoder(verbose=0, cols=None,
                                             mapping=None,
                                             drop_invariant=False,
                                             return_df=True,
                                             handle_unknown='value',
                                             handle_missing='value')

    cat_preprocessor = make_pipeline(binary_encoder)
    num_preprocessor = make_pipeline(StandardScaler())

    cat_transformed = cat_preprocessor.fit_transform(df_cat)

    num_transformed = num_preprocessor.fit_transform(df_num)
    num_transformed = pd.DataFrame(num_transformed, columns=num_cols)

    df = pd.concat((num_transformed, cat_transformed), axis=1)

    cols = df.columns

    return df, cols, num_preprocessor, cat_preprocessor


def fit_model(X, y):
    """
    Fits model to data
    """
    reg_params = 10.**np.linspace(-10, 5, 10)
    model = RidgeCV(alphas=reg_params, fit_intercept=True, cv=5)
    model.fit(X, y)

    return model


def training():
    print("Info: Step 3 - Training start ...")
    # Import cleaned data
    df = pd.read_csv('data-clean.csv')

    # Seperate features from target variable
    X = df.drop(['total_price', 'price_log'], axis=1)
    y = df['price_log']

    # X_2 = df.drop(['total_price', 'price_log'], axis=1)

    num_cols = ['accommodates', 'bathrooms', 'bedrooms', 'beds']
    cat_cols = ['zipcode', 'property_type', 'room_type', 'bed_type']

    # num_cols_simp = ['bedrooms', 'bathrooms']
    # cat_cols_simp = ['zipcode']

    # Pickles model,column list, and data preprocessors
    # Premium Model
    X, cols, num_preprocessor, cat_preprocessor = \
        transform_data(df=X, num_cols=num_cols, cat_cols=cat_cols)
    model = fit_model(X, y)

    # Remove old data if present
    file = pathlib.Path("model_v1.pk")
    if file.exists():
        print("Info: Removing model_v1.pk")
        os.remove("model_v1.pk")

    file = pathlib.Path("cols.pk")
    if file.exists():
        print("Info: Removing cols.pk")
        os.remove("cols.pk")

    file = pathlib.Path("num_preprocessor.pk")
    if file.exists():
        print("Info: Removing num_preprocessor.pk")
        os.remove("num_preprocessor.pk")

    file = pathlib.Path("cat_preprocessor.pk")
    if file.exists():
        print("Info: Removing cat_preprocessor.pk")
        os.remove("cat_preprocessor.pk")

    filename_model = 'model_v1.pk'
    with open('./'+filename_model, 'wb') as file:
        pickle.dump(model, file)

    filename_cols = 'cols.pk'
    with open('./'+filename_cols, 'wb') as file:
        pickle.dump(cols, file)

    file_name_numpre = 'num_preprocessor.pk'
    with open('./'+file_name_numpre, 'wb') as file:
        pickle.dump(num_preprocessor, file)

    file_name_catpre = 'cat_preprocessor.pk'
    with open('./'+file_name_catpre, 'wb') as file:
        pickle.dump(cat_preprocessor, file)

    print("Info: Training completed ...")