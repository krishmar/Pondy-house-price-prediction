############################################################
# Pondy House Price Prediction:                            #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                             #
# An end to End project for calculating House Price.       #
# Sart with feature engineering - Data cleaning/wrangling  #
# remove outliers, Hypertune parameters - build model      #
# Finally export the pretrianed model in to an pickle file #
############################################################

# Use following lines only if you need a clean build with no warnings
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import warnings
warnings.filterwarnings('ignore')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# modules for data modeling
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor
import pickle, json

# misc modules data cleaning / wrangling
import pandas as pd
import numpy as np


# create dataframe from available CSV file
df1 = pd.read_csv("pondy_house_prices.csv")
# print(df1.shape)

###############################################
# ~~~~~~~~~~~~~ Data Wrangling ~~~~~~~~~~~~~~~~
###############################################

# remove uneccessary (non-supportive) features
df2 = df1.drop(['area_type','society','balcony','availability'],axis='columns')
df3 = df2.dropna() # remove null data
# print(df3.shape)

# remove extra chracter in 'size' feature
df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
df4 = df3.copy() 

# clean 'total_sqft' feature with data in range (example:1200-1800) 
def convert_sqft_to_num(x):                  
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None
    
df4.total_sqft = df4.total_sqft.apply(convert_sqft_to_num)
df4 = df4[df4.total_sqft.notnull()]
df5 = df4.copy()
# print(df5.shape)


# create new column 'price_per_sqft' for price comparison
df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft'] 

# remove data in 'location' with lesser information 
df5.location = df5.location.apply(lambda x: x.strip())  
location_stats = df5.location.value_counts(ascending=False)
location_stats_less_than_10 = location_stats[location_stats<=10]
df5.location = df5.location.apply(lambda x:\
'other' if x in location_stats_less_than_10 else x)

# remove data with false BHK information.
df6 = df5[~(df5.total_sqft/df5.bhk < 300)]

###############################################
# ~~~~~~~~~~~~~ Feature Engineering ~~~~~~~~~~~
###############################################

# remove outliers found beyond "one-standard deviation"
def clean_pps_outliers(df):
    df_out = pd.DataFrame()
    for location, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft >=(m-st))
            & (subdf.price_per_sqft <=(m+st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

df7 = clean_pps_outliers(df6)
df7.shape


# remove outliers with mismatch in price and BHK
def clean_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, locationdf in df.groupby('location'):
        bhk_stats = { }
        for bhk, bhkdf in locationdf.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean':np.mean(bhkdf.price_per_sqft),
                'count':bhkdf.shape[0]
            }
        for bhk, bhkdf in locationdf.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(exclude_indices,\
                bhkdf[bhkdf.price_per_sqft < stats['mean']].index.values)
    return df.drop(exclude_indices, axis='index')

df8 = clean_bhk_outliers(df7)

# remove ouliers with mismatch in total bathroom and BHK informtion
df9 = df8[df8.bath < df8.bhk+2]

# finally remove non supportive features for model building
df10 = df9.drop(['size', 'price_per_sqft'],axis='columns')

# encoding location data with dummies function in pandas
dummies = pd.get_dummies(df10.location)
df11 = pd.concat([df10,dummies.drop('other',axis='columns')], axis='columns')
df12 = df11.drop('location', axis='columns')

###############################################
# ~~~~~~~~~~~~~ Data Modeling ~~~~~~~~~~~~~~~~
###############################################

# final data X, y for training and testing
X = df12.drop(['price'],axis='columns')
y = df12.price

# Hyper Parameter tunning with GridSearchCV
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
cross_val_score(LinearRegression(), X, y, cv=cv)

def hyper_parameter_tune(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], \
                           cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

print("Following are fine tune scores and their parameters using GridSearchcv\n")
print(hyper_parameter_tune(X,y))

# Choose LinearRegression as the best model with highest accuracy score
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)

# cross check build model
def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr.predict([x])[0]

Estimated_House_Price = predict_price('1st Phase JP Nagar',1000, 2, 2)

print("\n\nEstimated house price of '1st Phase JP Nagar',1000, 2, 2' is : ", Estimated_House_Price.round(2), " Lakhs")


##############################################################
# ~~~~~~~~ Exporting pretrained model for production ~~~~~~~~~
##############################################################


with open('model.pickle','wb') as f:
    pickle.dump(lr,f)


import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))
     

###############################################
# ~~~~~~~~~~~~~ Result ouput ~~~~~~~~~~~~~~~~
###############################################

# if everything working is fine, after executing the py file you should see 
# the following result in your console window screen.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Following are fine tune scores and their parameters using GridSearchcv

#                model  best_score                               best_params
# 0  linear_regression    0.847796                      {'normalize': False}
# 1              lasso    0.726738       {'alpha': 2, 'selection': 'cyclic'}
# 2      decision_tree    0.716117  {'criterion': 'mse', 'splitter': 'best'}


# Estimated house price of '1st Phase JP Nagar',1000, 2, 2 :  83.87  Lakhs

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~