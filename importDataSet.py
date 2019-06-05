#import liabrRY
import numpy as np
import pandas as pd
import matplotlib as plt

#IMPORT DATASETS
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#HANDEL MISSING DATA
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis= 0)# axis denotes 0 = columnwise, 1=rowise
imputer = imputer.fit(X[:, 1:3])
print(imputer)
X[:, 1:3] = imputer.transform(X[:, 1:3])#Fit to data, then transform it.
imputer1 = Imputer

# tranform categorical data
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_X = LabelEncoder()#LabelEncoder can be used to normalize labels.
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
#Encode categorical integer features as a one-hot numeric array.
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y)

#spliting the dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
