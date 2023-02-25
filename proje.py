import pandas as pd
import numpy as np

data = pd.read_csv(r"/content/megaGymDataset.csv")

data.head()

data.info()

data.describe()

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import matplotlib.pyplot as plt

data.hist(bins=50, figsize=(15,15))
plt.show()

missing_values = data.isnull().sum() 
missing_values

data['Desc'] = data.Desc.fillna(value=0)

data['Rating'] = data.Rating.fillna(value=0)

data['RatingDesc'] = data.RatingDesc.fillna(value=0)

data.isnull().sum()

data_Equipment_proximity = data[["Equipment"]]
data_Equipment_proximity.head(10)

from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
data_Equipment_encoded = ordinal_encoder.fit_transform(data_Equipment)
data_Equipment_encoded[:10]

ordinal_encoder.categories_

from sklearn.preprocessing import OneHotEncoder

onehot_encoder = OneHotEncoder()
data_Equipment_onehotencoder = onehot_encoder.fit_transform(data_Equipment_proximity)
data_Equipment_onehotencoder.toarray()

data.data_Equipment_proximity  = data_Equipment_encoded

x = data.drop("Rating", axis=1)
y = data.Rating

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(y,
                                                    x, 
                                                    test_size = 0.2, 
                                                    shuffle = False, 
                                                    random_state = None)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                  test_size = 0.25,
                                                  shuffle = False)

all = {"x train" : x_train,
       "x validation" : x_val,
       "x test" : x_test,
       "y train" : y_train,
       "y validation": y_val,
       "y test": y_test}

for i in all:
    print(f"{i} satır sayısı: {len(all.get(i))}")

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt

def evaluate_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    print(f"Random Forest RMSE: {rmse:.2f}")
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    print(f"Linear Regression RMSE: {rmse:.2f}")
    
    from sklearn.svm import SVR

    svm = SVR(kernel="linear")
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    print(f"SVM RMSE: {rmse:.2f}")

evaluate_models(x, y)
