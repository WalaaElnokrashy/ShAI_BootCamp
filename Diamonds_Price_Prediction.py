#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

D_Train=pd.read_csv("dataset/train.csv")
D_Train.drop(labels=['Id'], axis=1,inplace=True)

D_Train.head()

D_Train.shape

D_Train.info()

D_Train.describe()

D_Train.median(numeric_only= True)

D_Train.isnull().sum()


D_Train.isna().sum().sum()

D_Train.drop_duplicates(inplace= True)


cat_cols= D_Train.columns[1:4]
num_cols= D_Train.columns[4:].insert(0, D_Train.columns[0])
print(cat_cols)
print(num_cols)


for cat_col in cat_cols:
    print(cat_col+ "\n"+ str(D_Train[cat_col].value_counts(normalize= True) * 100) + "\n")

fig, axes = plt.subplots(2, 2, figsize=(8,8))
c,r= 0, 0
for cat_col in cat_cols:
    x= D_Train[cat_col].value_counts()

    axes[r, c].pie(x, labels= x.index, autopct= '%1.1f%%')
    axes[r, c].set_title(cat_col)
    c= c + 1
    if c == 2:
        r= r + 1
        c= 0

    
plt.show()


df_corr= D_Train.corr()
sns.heatmap(df_corr, annot= True)
plt.show()



sns.pairplot(D_Train)
plt.show()


D_Train.hist(bins= 70, figsize=(15, 10))
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(12,10))
r, c =0, 0
for cat_col in cat_cols:
    sns.boxplot(ax=axes[r, c], data=D_Train, y='price', x= D_Train[cat_col])
    c= c + 1
    if c == 2:
        r= r + 1
        c= 0
    
plt.show()


fig, axes = plt.subplots(3, 3, figsize=(12,10))
r, c =0, 0
for num_col in num_cols:
    sns.boxplot(ax=axes[r, c], x= D_Train[num_col] )
    c= c + 1
    if c == 3:
        r= r + 1
        c= 0
    
plt.show()


# Handel outliers

print(f"rows= {D_Train.shape[0]}")
zero_df= D_Train[(D_Train['x'] == 0) | (D_Train['y'] == 0) | (D_Train['z'] <= 1.07)]
D_Train.drop(zero_df.index, inplace= True)
print(f"rows= {D_Train.shape[0]}")

D_Train.nlargest(5, "x")


D_Train.nlargest(5, "y")


D_Train.nlargest(5, "z")


print(f"rows= {D_Train.shape[0]}")
largest_df= D_Train[(D_Train['x'] == 10.74) | (D_Train['y'] >= 31.80) | (D_Train['z'] == 31.80)]
D_Train.drop(largest_df.index, inplace= True)
print(f"rows= {D_Train.shape[0]}")

sns.pairplot(D_Train[['x', 'y', 'z', 'price']])
plt.show()


#  Combining x,y,z with one column called size

D_Train['size']=D_Train['x']*D_Train['y']*D_Train['z']
D_Train.drop(labels=['x','y','z'], axis=1,inplace=True)
D_Train.head()

D_Train_num =D_Train.select_dtypes(include=[np.number])


# Getting Standard Scaler

from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
D_Train_num_std_scaled = std_scaler.fit_transform(D_Train_num)
print(D_Train_num_std_scaled[:,0])

D_Train_cat=D_Train[["cut",'color',"clarity"]] 


# Getting OrdinalEncoder to encode categorical values

from sklearn.preprocessing import OrdinalEncoder
categories = [
    ['Ideal','Premium','Very Good','Good','Fair'],
    ['D','E','F','G','H','I','J'],
    ['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2','I1']
]

ordinal_encoder = OrdinalEncoder(categories=categories)
D_Train_encoded=ordinal_encoder.fit_transform(D_Train_cat)

print(ordinal_encoder.categories_)

# Splitting the data into test and train sets
from sklearn.model_selection import train_test_split
train_set,test_set= train_test_split(D_Train, test_size=0.2, random_state=42)

# Putting the modifications we made to the data into a pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

Diamond=train_set.drop("price",axis=1)
Diamond_labels=train_set['price'].copy()
X_test=test_set.drop("price",axis=1)
Y_test=test_set['price'].copy()
Diamond_num = Diamond.drop(['cut','color','clarity'], axis=1)
num_attribs=list(Diamond_num)
cat_attribs = ['cut','color','clarity']
num_pipeline = Pipeline([
 ('std_scaler', StandardScaler())])

full_pipeline = ColumnTransformer([
 ("num", num_pipeline, num_attribs),
 ("cat", ordinal_encoder, cat_attribs)])
Diamond_prepared = full_pipeline.fit_transform(Diamond)


# Selecting a model and train it.

# Linear Regression

rmse_all= {}

from sklearn.linear_model import LinearRegression
lr_model= make_pipeline(full_pipeline, LinearRegression())
lr_model.fit(Diamond, Diamond_labels)
y_pred_lr= lr_model.predict(X_test)
lr_rmse = mean_squared_error(Y_test, y_pred_lr, squared=False)
rmse_all["lr"] = lr_rmse
lr_rmse


# Polynomial Regression

from sklearn.preprocessing import PolynomialFeatures
poly_features= make_pipeline(full_pipeline, PolynomialFeatures(degree = 2))
X_poly= poly_features.fit_transform(Diamond, Diamond_labels)
poly_model= LinearRegression()
poly_model.fit(X_poly, Diamond_labels)

poly_test= make_pipeline(full_pipeline, PolynomialFeatures(degree = 2))
x_test_poly= poly_test.fit_transform(X_test)

y_pred_poly= poly_model.predict(x_test_poly)
poly_rmse = mean_squared_error(Y_test, y_pred_poly, squared=False)
rmse_all["poly"] = poly_rmse
poly_rmse


# SVM
from sklearn.svm import SVR 

svr_model = make_pipeline(full_pipeline,SVR(kernel='poly', degree= 2))
svr_model.fit(Diamond, Diamond_labels)
y_pred_svr=svr_model.predict(X_test)
svr_rmse = mean_squared_error(Y_test, y_pred_svr, squared=False)
rmse_all["svr"] = svr_rmse
svr_rmse

# Random Forest
from sklearn.ensemble import RandomForestRegressor
rand_forest_model = make_pipeline(full_pipeline,RandomForestRegressor(n_estimators=100, random_state=42))
rand_forest_model.fit(Diamond, Diamond_labels)
y_pred_rand_forest=rand_forest_model.predict(X_test)
rand_forest_rmse = mean_squared_error(Y_test, y_pred_rand_forest, squared=False)
rmse_all["rand_forest"] = rand_forest_rmse
rand_forest_rmse

# XGboost
XGB_model = make_pipeline(full_pipeline,XGBRegressor())
XGB_model.fit(Diamond, Diamond_labels)
y_pred_XGB=XGB_model.predict(X_test)
xgb_rmse = mean_squared_error(Y_test, y_pred_XGB, squared=False)
rmse_all["XGboost"] = xgb_rmse
xgb_rmse
rmse_all

# Fine-Tuning the model's hyperparameters
param_grid={
    'XGB__max_depth': [1, 5, None],
    'XGB__reg_alpha': [0, 50,100,150],
    'XGB__reg_lambda': [.15,.20,.10,0],
    'XGB__n_estimators': [10,50,100,150,200],
    'XGB__learning_rate': [.15,.17]
}

XGboost= Pipeline([
 ("preprocessing", full_pipeline),
 ("XGB", XGBRegressor()),
])

grid_search = GridSearchCV(XGboost, param_grid, cv=5,scoring='neg_root_mean_squared_error',return_train_score = True,n_jobs=-1)
grid_search.fit(Diamond, Diamond_labels)
grid_search.best_params_
final_model = grid_search.best_estimator_
y_pred_XGB=final_model.predict(X_test)
rf_rmse = mean_squared_error(Y_test, y_pred_XGB, squared=False)
rf_rmse

# Creating the submisson
df_test=pd.read_csv('dataset/test.csv')
df_test['size']=df_test['x']*df_test['y']*df_test['z']
df_test.drop(labels=['x','y','z'], axis=1,inplace=True)
df_test.head()
id_test= df_test['Id']
df_test.drop(labels= ['Id'], axis=1,inplace=True)
y_pred= final_model.predict(df_test)
submission = pd.DataFrame({"id": id_test, "price": y_pred})
submission.to_csv('submission.csv',index=None)
