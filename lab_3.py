import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# read and print the first few rows of the data
data = pd.read_csv('src/epi_r.csv')
print(data.head())
print('The shape of the data is:', data.shape, '\n')
print('First 50 columns: ', data.columns[:50], '\n')
print('Last 50 columns: ', data.columns[-50:], '\n')
print('Number of numerical feature (float):', len([0 for col in data.columns if data[col].dtype==float]) )

most_important_features = [col for col in data.drop('title', axis=1).columns if data[col].sum()>5000]
#5000 is an arbitrary number

print('Most important features: ', most_important_features)

X = data[most_important_features].copy()
print(X.head())
print(X.isnull().sum())
all_nan = X[['calories', 'protein', 'fat', 'sodium']].isnull().all(axis=1)
print('Number of rows without calories, protein, fat and sodium: ', sum(all_nan))

from sklearn.impute import SimpleImputer

X.drop(X.loc[all_nan].index, axis=0, inplace=True)

imp = SimpleImputer()
cols = X.columns
X = pd.DataFrame(imp.fit_transform(X))
X.columns = cols 

X['calories'].replace(0, X['calories'].mean(), inplace=True)

print('Number of missing values: ', X.isnull().sum().sum())

y = X['rating']
X = X.drop('rating', axis=1)

X['fat/calories'] = X['fat'] / X['calories']
X['protein/calorires'] = X['protein'] / X['calories']
from sklearn.feature_selection import mutual_info_regression

mi_scores = mutual_info_regression(X, y)


plt.barh(X.columns, mi_scores)
plt.show()

X.drop(['vegetarian', 'peanut free', 'soy free', 'kosher'], axis=1, inplace=True)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(pred)
err = mean_squared_error(pred, y_test)

print('MSE: ', err)