import numpy as np
import pandas as pd
import dill as pickle
import math
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import category_encoders as ce
from sklearn.linear_model import RidgeCV

import warnings
warnings.filterwarnings("ignore")

model = None
cols = None
num_preprocessor = None
cat_preprocessor = None


def load_pickle_files():
    print("Info: Step 4 - Load Pickle Files start ...")
    global model, cols
    global num_preprocessor, cat_preprocessor

    """ Python file containing predict function """
    with open('./model_v1.pk', 'rb') as f:
        model = pickle.load(f)

    with open('./cols.pk', 'rb') as f:
        cols = pickle.load(f)

    with open('./num_preprocessor.pk', 'rb') as f:
        num_preprocessor = pickle.load(f)

    with open('./cat_preprocessor.pk', 'rb') as f:
        cat_preprocessor = pickle.load(f)
    print("Info: Load Pickle Files completed ...")


def get_prediction(zipcode, property_type, room_type, accommodates,
                   bathrooms, bedrooms, beds, bed_type):
    global model, cols
    global num_preprocessor, cat_preprocessor

    data = {"zipcode": zipcode,
            "property_type": property_type,
            "room_type": room_type,
            "accommodates": accommodates,
            "bathrooms": bathrooms,
            "bedrooms": bedrooms,
            "beds": beds,
            "bed_type": bed_type}

    # Create dataframe from JSON dict
    data = pd.DataFrame.from_dict([data])

    num_cols = ['accommodates', 'bathrooms', 'bedrooms', 'beds']
    cat_cols = ['zipcode', 'property_type', 'room_type', 'bed_type']

    # Seperate into Numeric and Categorical columns
    df_num = data[num_cols]
    df_cat = data[cat_cols]

    # Use train data preprocessor
    cat_transformed = cat_preprocessor.transform(df_cat)

    # Use train data preprocessor
    num_transformed = num_preprocessor.transform(df_num)
    num_transformed = pd.DataFrame(num_transformed, columns=num_cols)

    # Concatenate numeric and categorical dataframes
    df_transformed = pd.concat((num_transformed, cat_transformed), axis=1)
    # Create blank dataframe using columns from transformed train data
    df_blank = pd.DataFrame(columns=cols)

    # Concatenate
    df_model = pd.concat((df_blank, df_transformed))
    df_model = df_model.replace(np.nan, 0)

    y_pred = model.predict(df_model)
    prediction = int(math.exp(y_pred[0]))

    return prediction