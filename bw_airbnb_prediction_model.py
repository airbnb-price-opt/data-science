"""BW Airbnb Prediction Model.ipynb
"""

#Regular imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import io

#SKLearn imports
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

#Imputer and Encoders
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb

#extras
from geopy.distance import great_circle
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
boost = xgb.XGBRegressor()
clf = DecisionTreeRegressor()

#display more rows and columns than the default
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#import the files from files
from google.colab import files

uploaded = files.upload()

#our df
listings = pd.read_csv(io.BytesIO(uploaded['listings.csv'])) 

#our df2
list_sum = pd.read_csv(io.BytesIO(uploaded['listings_summary.csv']))

df_cols = ["room_type","number_of_reviews","calculated_host_listings_count","availability_365","longitude","latitude","price"]
df = listings[df_cols]


df2_cols = ["cancellation_policy","host_identity_verified","amenities","security_deposit",
                "cleaning_fee","guests_included","extra_people","review_scores_rating",
               "bathrooms","bedrooms","beds","bed_type","accommodates", "description"]
df2 = list_sum[df2_cols]

#making distance column and removing lat and long
def distance_from_berlin(lat, lon):
    berlin_centre = (52.5027778, 13.404166666666667)
    record = (lat, lon)
    return great_circle(berlin_centre, record).km

#add distanse dataset
df['distance'] = df.apply(lambda x: distance_from_berlin(x.latitude, x.longitude), axis=1)

del df['latitude']
del df['longitude']

#turning room_type to numerical values and categories
def set_room_type(x):
    if x=="Private room":
        return 1
    else:
        return 0
    
df["room_type"] = list(map(set_room_type, df["room_type"]))
df["room_type"] = df["room_type"].astype("category")

#cleaning NaNs
#they have to have at least 1 and not many of these nans
df2.bathrooms = df2.bathrooms.fillna(1.0)
df2.bedrooms = df2.bedrooms.fillna(1.0)
df2.beds = df2.beds.fillna(1.0)

#turning to numerical and categorical
beds = {"Real Bed":0, "Pull-out Sofa":1, "Futon":2, "Couch":3, "Airbed":4}

def bed_types(n):
    return beds[n]

df2.bed_type = list(map(bed_types, df2.bed_type))
df2.bed_type = df2.bed_type.astype("category")

#turning to numerical and categorical
canc = {"flexible":0, "moderate":1, "strict_14_with_grace_period":2, "super_strict_30":3, "super_strict_60":4}

def canc_pol(n):
    return canc[n]

df2.cancellation_policy = list(map(canc_pol, df2.cancellation_policy))
df2.cancellation_policy = df2.cancellation_policy.astype("category")

#switching values from objects to int
df2.host_identity_verified.fillna("f", inplace=True)

host = {"f":0, "t":1}

def host_iden(n):
    return host[n]

df2.host_identity_verified = list(map(host_iden, df2.host_identity_verified))
df2.host_identity_verified = df2.host_identity_verified.astype("category")

df2.host_identity_verified = [1 if i=='t' else 0 for i in df2.host_identity_verified ]
df2.host_identity_verified = df2.host_identity_verified.astype("category")

#create new DataFrame with amnenities
amenities_df = list_sum[['id', 'amenities']]

from collections import Counter
collection = Counter()
amenities_df['amenities'].str.strip('{}').str.replace('"', '')\
               .str.split(',').apply(collection.update)

# Choosing only the 45 amenities most popular from a total of 136
list_ame = collection.most_common(40)
list_amenities = []
for i in list_ame:
    list_amenities.append(i[0])
    
#create a column for every amenitie
for i in list_amenities:
    amenities_df[i] = amenities_df['amenities'].str.contains(i)
    
#drop column 'amenities'
amenities_df.drop('amenities', inplace=True, axis=1)

#transform DF boolean dtype columns in integer dtype 
amenities_df = amenities_df.astype(int)

#create column number of amenities 'amenities_num' in DF listings 
listings['amenities_num'] = amenities_df.drop(
    columns=['id', 'translation missing: en.hosting_amenity_49']).sum(axis=1)

#inserting new amenities column and dropping old one
df2.insert(1,'amenities_num', listings['amenities_num'])

df2.drop('amenities', inplace=True, axis=1)

#next few code cells strip the $ sign from the values
df2.extra_people =  list(map(lambda x: float(str(x).replace(',','').replace('$','')),df2.extra_people)) 
df2.extra_people = df2.extra_people.astype(float)

df2.extra_people =  list(map(lambda x: float(str(x).replace(',','').replace('$','')),df2.extra_people)) 
df2.extra_people = df2.extra_people.astype(float)

df2.security_deposit = list(map(lambda x: float(str(x).replace(',','').replace('$','')), df2.security_deposit))
df2.cleaning_fee = list(map(lambda x: float(str(x).replace(',','').replace('$','')), df2.cleaning_fee))

#formula for security_deposit median value
val = df2.security_deposit.value_counts(dropna=True)
indx = df2.security_deposit.value_counts(dropna=True).index

secd_med = (sum(val*indx))/sum(df2.security_deposit.value_counts(dropna=True)[0:2000])
secd_med = float(round(secd_med,2))

#filling these NaNs
df2.security_deposit = df2.security_deposit.fillna(secd_med)

#formula for cleaning_fee median value
val = df2.cleaning_fee.value_counts(dropna=True)
indx = df2.cleaning_fee.value_counts(dropna=True).index

clean_med = (sum(val*indx))/sum(df2.cleaning_fee.value_counts(dropna=True)[0:2000])
clean_med = float(round(clean_med,2))

#filling these NaNs
df2.cleaning_fee = df2.cleaning_fee.fillna(clean_med)

#formula for review_score median value
val = df2.review_scores_rating.value_counts(dropna=True)
indx = df2.review_scores_rating.value_counts(dropna=True).index

rev_med = (sum(val*indx))/sum(df2.review_scores_rating.value_counts(dropna=True))
rev_med = float(round(rev_med,2))

#filling these NaNs
df2.review_scores_rating = df2.review_scores_rating.fillna(rev_med)

#feature engineering to create size column from description column
df2['size'] = df2['description'].str.extract('(\d{2,3}\s?[smSM])', expand=True)
df2['size'] = df2['size'].str.replace("\D", "")

# change datatype of size into float
df2['size'] = df2['size'].astype(float)

print('NaNs in size_column absolute:     ', df2['size'].isna().sum())
print('NaNs in size_column in percentage:', round(df2['size'].isna().sum()/len(df2),3), '%')

df2.drop(['description'], axis=1, inplace=True)

sub_df = df2[['accommodates', 'bathrooms', 'bedrooms', 'cleaning_fee', 
                 'security_deposit', 'extra_people', 'size']]

# split datasets
train_data = sub_df[sub_df['size'].notnull()]
test_data  = sub_df[sub_df['size'].isnull()]

# define X
X_train = train_data.drop('size', axis=1)
X_test  = test_data.drop('size', axis=1)

# define y
y_train = train_data['size']

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_test = linreg.predict(X_test)

y_test = pd.DataFrame(y_test)
y_test.columns = ['size']

# make the index of X_test to an own dataframe
prelim_index = pd.DataFrame(X_test.index)
prelim_index.columns = ['prelim']

# ... and concat this dataframe with y_test
y_test = pd.concat([y_test, prelim_index], axis=1)
y_test.set_index(['prelim'], inplace=True)
y_test.head()

new_test_data = pd.concat([X_test, y_test], axis=1)
new_test_data.head()

sub_df_new = pd.concat([new_test_data, train_data], axis=0)

print(sub_df_new.shape)
sub_df_new.head()

df2.drop('size', inplace=True, axis=1)

df2.insert(1,'size', sub_df_new['size'])
df2.head(2)

data = pd.concat([df, df2], axis=1)
data.head(2)

data.drop(data[ (data['size'] == 0.) | (data['size'] > 300.) ].index, axis=0, inplace=True)
data.drop(data[ (data.price > 400) | (data.price == 0) ].index, axis=0, inplace=True)

"""# Model"""

target = data["price"]

values = data.drop(["price"], axis=1)

X = values
y = target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)

#looking for best parameters
param_grid = {'n_estimators': [100, 150, 200],
              'learning_rate': [0.01, 0.05, 0.1], 
              'max_depth': [3, 4, 5, 6, 7],
              'colsample_bytree': [0.6, 0.7, 1],
              'gamma': [0.0, 0.1, 0.2]}

booster_grid_search = GridSearchCV(boost, param_grid, cv=3, n_jobs=-1)
booster_grid_search.fit(X_train, y_train)

print(booster_grid_search.best_params_)

#best_params_ = colsample_bytree': 0.6, 'gamma': 0.2, 'learning_rate': 0.05, 'max_depth': 7, 'n_estimators': 200
#tuning our parameters

boost = xgb.XGBRegressor(colsample_bytree=0.7, gamma=0.2, learning_rate=0.05, 
                           max_depth=7, n_estimators=200, random_state=4)

# train
boost.fit(X_train, y_train)

# predict
y_pred_train = boost.predict(X_train)
y_pred_test = boost.predict(X_test)

#final RMSE and r2 scores
RMSE = np.sqrt(mean_squared_error(y_test, y_pred_test))
print(f"RMSE: {round(RMSE, 4)}")

r2 = r2_score(y_test, y_pred_test)
print(f"r2: {round(r2, 4)}")