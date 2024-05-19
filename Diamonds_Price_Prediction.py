#!/usr/bin/env python
# coding: utf-8

# ## Columns
# - price price in US dollars (\$326--\$18,823) 
# - carat weight of the diamond (0.2--5.01)
# - cut quality of the cut (Fair, Good, Very Good, Premium, Ideal)
# - color diamond colour, from J (worst) to D (best)
# - clarity a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))
# - x length in mm (0--10.74)
# - y width in mm (0--58.9)
# - z depth in mm (0--31.8)
# - depth total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43--79)
# - table width of top of diamond relative to widest point (43--95)

# # Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor


# # Reading the data

# In[2]:


D_Train=pd.read_csv("dataset/train.csv")
D_Train.drop(labels=['Id'], axis=1,inplace=True)


# In[3]:


D_Train.head()


# <div style="float:left">
#     <h3> carat </h3>
# <img src= "https://cdn.loosegrowndiamond.com/wp-content/uploads/2022/02/different-carat-sizes.png" alt="carat image" width="450" height="500"></div>
# 
# <div style="float:right">
#     <h3> cut </h3>
# <img src= "https://cdn.shopify.com/s/files/1/0403/0762/2040/files/diamond-chart-cut_2_1024x1024.jpg?v=1617401955" alt="cut image" width="450" height="500"></div>
# 
# <div style="float:left">
#     <h3> clarity </h3>
# <img src= "https://assets-rarecarat.s3.amazonaws.com/images/blog/clarity-chart.jpg" alt="cut image" width="450" height="500"></div>
# 
# <div style="float:right">
#     <h3> color </h3>
# <img src= "https://www.brides.com/thmb/GBtDWdJwsYmu17LqrVGm2lR49nU=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/diamond-color-chart-5093397_horizontal-b8d3872096fd47c78d244d40cc920099.png" alt="color image" width="450" height="500"></div>
# 

# ### Take a Quick Look at the Data Structure

# In[4]:


D_Train.shape


# > We have `43152` records and `11` Features

# In[5]:


D_Train.info()


# > normal data types and the data doesn't have any null values

# In[6]:


D_Train.describe()


# > abnormal values in `x`,`y` and `z` columns min = 0

# In[7]:


D_Train.median(numeric_only= True)


# > may be there are some outliers values in `price` column

# # Cleaning the data (null and duplicated values)

# In[8]:


D_Train.isnull().sum()


# In[9]:


D_Train.isna().sum().sum()


# In[10]:


D_Train.drop_duplicates(inplace= True)


# In[11]:


cat_cols= D_Train.columns[1:4]
num_cols= D_Train.columns[4:].insert(0, D_Train.columns[0])
print(cat_cols)
print(num_cols)


# In[12]:


for cat_col in cat_cols:
    print(cat_col+ "\n"+ str(D_Train[cat_col].value_counts(normalize= True) * 100) + "\n")


# In[13]:


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


# > the data is unbalanced, <br>Ideal cut is 39.87% of `cut` column and fair is only 3%, <br>G is 21.00 % of `color` column and j is only 5.31%<br>SI1 is 24.17% of `clarity` column and I1 is only 1.40%

# # Discover and Visualize the Data to Gain Insights

# In[14]:


df_corr= D_Train.corr()
sns.heatmap(df_corr, annot= True)
plt.show()


# > there are a strong linear relation between (`carat`, `x`, `y`, `z`) columns and `price` column

# In[15]:


sns.pairplot(D_Train)
plt.show()


# > some columns values take curved shape

# In[16]:


D_Train.hist(bins= 70, figsize=(15, 10))
plt.show()


# > the data has outliers. `depth`, `table`, `y` and `x` columns are normaly distributed. other columns is positive (left) skewed

# In[17]:


fig, axes = plt.subplots(2, 2, figsize=(12,10))
r, c =0, 0
for cat_col in cat_cols:
    sns.boxplot(ax=axes[r, c], data=D_Train, y='price', x= D_Train[cat_col])
    c= c + 1
    if c == 2:
        r= r + 1
        c= 0
    
plt.show()


# In[18]:


fig, axes = plt.subplots(3, 3, figsize=(12,10))
r, c =0, 0
for num_col in num_cols:
    sns.boxplot(ax=axes[r, c], x= D_Train[num_col] )
    c= c + 1
    if c == 3:
        r= r + 1
        c= 0
    
plt.show()


# ### Handel outliers

# In[19]:


print(f"rows= {D_Train.shape[0]}")
zero_df= D_Train[(D_Train['x'] == 0) | (D_Train['y'] == 0) | (D_Train['z'] <= 1.07)]
D_Train.drop(zero_df.index, inplace= True)
print(f"rows= {D_Train.shape[0]}")


# In[20]:


D_Train.nlargest(5, "x")


# In[21]:


D_Train.nlargest(5, "y")


# In[22]:


D_Train.nlargest(5, "z")


# In[23]:


print(f"rows= {D_Train.shape[0]}")
largest_df= D_Train[(D_Train['x'] == 10.74) | (D_Train['y'] >= 31.80) | (D_Train['z'] == 31.80)]
D_Train.drop(largest_df.index, inplace= True)
print(f"rows= {D_Train.shape[0]}")


# In[24]:


sns.pairplot(D_Train[['x', 'y', 'z', 'price']])
plt.show()


# > also x, y, z columns are curved

# # Prepare the data for ML.  

# ## Combining x,y,z with one column called size

# In[25]:


D_Train['size']=D_Train['x']*D_Train['y']*D_Train['z']
D_Train.drop(labels=['x','y','z'], axis=1,inplace=True)
D_Train.head()


# In[26]:


D_Train_num =D_Train.select_dtypes(include=[np.number])


# ## Getting Standard Scaler

# In[27]:


from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
D_Train_num_std_scaled = std_scaler.fit_transform(D_Train_num)
print(D_Train_num_std_scaled[:,0])


# In[28]:


D_Train_cat=D_Train[["cut",'color',"clarity"]] 


# ## Getting OrdinalEncoder to encode categorical values

# In[29]:


from sklearn.preprocessing import OrdinalEncoder
categories = [
    ['Ideal','Premium','Very Good','Good','Fair'],
    ['D','E','F','G','H','I','J'],
    ['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2','I1']
]

ordinal_encoder = OrdinalEncoder(categories=categories)
D_Train_encoded=ordinal_encoder.fit_transform(D_Train_cat)


# In[30]:


print(ordinal_encoder.categories_)


# ## Splitting the data into test and train sets

# In[31]:


from sklearn.model_selection import train_test_split
train_set,test_set= train_test_split(D_Train, test_size=0.2, random_state=42)


# ## Putting the modifications we made to the data into a pipeline

# In[32]:


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


# # Select a model and train it.

# # Linear Regression

# In[33]:


rmse_all= {}


# In[34]:


from sklearn.linear_model import LinearRegression

lr_model= make_pipeline(full_pipeline, LinearRegression())
lr_model.fit(Diamond, Diamond_labels)


# In[35]:


y_pred_lr= lr_model.predict(X_test)
lr_rmse = mean_squared_error(Y_test, y_pred_lr, squared=False)
rmse_all["lr"] = lr_rmse
lr_rmse


# ## Polynomial Regression

# In[36]:


from sklearn.preprocessing import PolynomialFeatures

poly_features= make_pipeline(full_pipeline, PolynomialFeatures(degree = 2))
X_poly= poly_features.fit_transform(Diamond, Diamond_labels)


# In[37]:


poly_model= LinearRegression()
poly_model.fit(X_poly, Diamond_labels)


# In[38]:


poly_test= make_pipeline(full_pipeline, PolynomialFeatures(degree = 2))
x_test_poly= poly_test.fit_transform(X_test)

y_pred_poly= poly_model.predict(x_test_poly)
poly_rmse = mean_squared_error(Y_test, y_pred_poly, squared=False)
rmse_all["poly"] = poly_rmse
poly_rmse


# ## SVM

# In[40]:


from sklearn.svm import SVR 

svr_model = make_pipeline(full_pipeline,SVR(kernel='poly', degree= 2))
svr_model.fit(Diamond, Diamond_labels)


# In[41]:


y_pred_svr=svr_model.predict(X_test)
svr_rmse = mean_squared_error(Y_test, y_pred_svr, squared=False)
rmse_all["svr"] = svr_rmse
svr_rmse


# ## Random Forest

# In[42]:


from sklearn.ensemble import RandomForestRegressor

rand_forest_model = make_pipeline(full_pipeline,RandomForestRegressor(n_estimators=100, random_state=42))
rand_forest_model.fit(Diamond, Diamond_labels)


# In[43]:


y_pred_rand_forest=rand_forest_model.predict(X_test)
rand_forest_rmse = mean_squared_error(Y_test, y_pred_rand_forest, squared=False)
rmse_all["rand_forest"] = rand_forest_rmse
rand_forest_rmse


# ## XGboost

# In[44]:


XGB_model = make_pipeline(full_pipeline,XGBRegressor())
XGB_model.fit(Diamond, Diamond_labels)


# In[45]:


y_pred_XGB=XGB_model.predict(X_test)
xgb_rmse = mean_squared_error(Y_test, y_pred_XGB, squared=False)
rmse_all["XGboost"] = xgb_rmse
xgb_rmse


# In[46]:


rmse_all


# > XGboost has the lowest RMSE (the best model)

# # Fine-Tuning the model's hyperparameters

# In[47]:


param_grid={
    'XGB__max_depth': [1, 5, None],
    'XGB__reg_alpha': [0, 50,100,150],
    'XGB__reg_lambda': [.15,.20,.10,0],
    'XGB__n_estimators': [10,50,100,150,200],
    'XGB__learning_rate': [.15,.17]
}


# In[48]:


XGboost= Pipeline([
 ("preprocessing", full_pipeline),
 ("XGB", XGBRegressor()),
])

grid_search = GridSearchCV(XGboost, param_grid, cv=5,scoring='neg_root_mean_squared_error',return_train_score = True,n_jobs=-1)
grid_search.fit(Diamond, Diamond_labels)


# In[49]:


grid_search.best_params_


# In[50]:


final_model = grid_search.best_estimator_


# In[51]:


y_pred_XGB=final_model.predict(X_test)
rf_rmse = mean_squared_error(Y_test, y_pred_XGB, squared=False)
rf_rmse


# # Creating the submisson

# In[52]:


df_test=pd.read_csv('dataset/test.csv')
df_test['size']=df_test['x']*df_test['y']*df_test['z']
df_test.drop(labels=['x','y','z'], axis=1,inplace=True)
df_test.head()


# In[53]:


id_test= df_test['Id']
df_test.drop(labels= ['Id'], axis=1,inplace=True)


# In[54]:


y_pred= final_model.predict(df_test)


# In[55]:


submission = pd.DataFrame({"id": id_test, "price": y_pred})


# In[56]:


submission.to_csv('submission.csv',index=None)

