# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 20:37:44 2019

@author: nmaharan
"""
# =============================================================================
# Importing required packages initial
# =============================================================================
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Reading The dataset and store it in a dataframe
# =============================================================================

credit_df = pd.read_csv(r"D:\SapientFirstRound\Complete-Data-Set\application_train.csv");

# =============================================================================
# Basic Information like shape, size and null values
# =============================================================================

#Sample Dataset
credit_df.head()

#Basic oinformation about dataset
credit_df.info()

#sample coulumns
credit_df.columns

#Shape of the dataset
credit_df.shape

# Decsription about dataset
credit_df.describe()

#Printing null sum of columns
credit_df.isnull().sum()

#Getting Duplicate values:
credit_df.duplicated(subset=None,keep='first').sum()

# =============================================================================
# 
# =============================================================================

f,ax=plt.subplots(1,2,figsize=(18,8))
credit_df['TARGET'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Loan Sanctioned')
ax[0].set_ylabel('')
sns.countplot('Loan Sanctioned',credit_df=credit_df,ax=ax[1])
ax[1].set_title('Loan Sanctioned')
plt.show()

# =============================================================================
# First Preprocessing by panada profiling to get insights 
# of data from the html file
# =============================================================================

import pandas_profiling
credit_df_profile = pandas_profiling.ProfileReport(credit_df)
credit_df_profile.to_file(outputfile="Finacial_Credit_Preprocess.html")

# =============================================================================
# Drop columns which has more than 50% of null values
# =============================================================================
credit_df_clean = credit_df.drop(['CNT_FAM_MEMBERS','DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE',
                            'DEF_60_CNT_SOCIAL_CIRCLE','OBS_30_CNT_SOCIAL_CIRCLE','FLOORSMIN_MEDI',
                            'FLOORSMIN_MODE','LANDAREA_MEDI','LANDAREA_MODE','LIVINGAPARTMENTS_AVG',
                            'LIVINGAPARTMENTS_MEDI','LIVINGAPARTMENTS_MODE','LIVINGAREA_AVG','LIVINGAREA_MEDI',
                            'LIVINGAREA_MODE','AMT_GOODS_PRICE','APARTMENTS_MEDI','APARTMENTS_MODE','BASEMENTAREA_MEDI',
                            'BASEMENTAREA_MODE','COMMONAREA_MEDI','COMMONAREA_MODE','ELEVATORS_MEDI','ELEVATORS_MODE',
                            'ENTRANCES_MEDI','ENTRANCES_MODE','FLAG_MOBIL','FLOORSMAX_MEDI','FLOORSMAX_MODE',
                            'NONLIVINGAPARTMENTS_MEDI','NONLIVINGAPARTMENTS_MODE','NONLIVINGAREA_MEDI','NONLIVINGAREA_MODE',
                            'REGION_RATING_CLIENT_W_CITY','TOTALAREA_MODE','YEARS_BEGINEXPLUATATION_MEDI',
                            'YEARS_BEGINEXPLUATATION_MODE','YEARS_BUILD_MEDI','YEARS_BUILD_MODE',
                            'YEARS_BEGINEXPLUATATION_AVG','BASEMENTAREA_AVG','COMMONAREA_AVG','ELEVATORS_AVG',
                            'EMERGENCYSTATE_MODE','ENTRANCES_AVG','FLOORSMIN_AVG','HOUSETYPE_MODE',
                            'FONDKAPREMONT_MODE','EXT_SOURCE_1','EXT_SOURCE_3','FLOORSMAX_AVG',
                            'NONLIVINGAPARTMENTS_AVG','NONLIVINGAREA_AVG','WALLSMATERIAL_MODE',
                            'LANDAREA_AVG','YEARS_BUILD_AVG','AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_MON',
                            'AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_QRT',
                            'AMT_REQ_CREDIT_BUREAU_YEAR','ORGANIZATION_TYPE','OCCUPATION_TYPE','NAME_EDUCATION_TYPE',
                            'NAME_TYPE_SUITE'],axis=1)



# =============================================================================
# Fill nulls and missing values in below 
# =============================================================================
# Fill with 0 as the person has not have any car 
credit_df_clean['OWN_CAR_AGE'].fillna(0, inplace=True)
# Fill the amount with 0  
credit_df_clean['AMT_ANNUITY'].fillna(0, inplace=True)
# Fill with 0 for the apartment avg also
credit_df_clean['APARTMENTS_AVG'].fillna(0, inplace=True)
# Fill with 0 for the external sources also
credit_df_clean['EXT_SOURCE_2'].fillna(0, inplace = True)

# =============================================================================
# Checking for null values after cleansing of data
# =============================================================================

credit_df_clean.columns[credit_df_clean.isnull().any()]

# No futher null values so processing further

# =============================================================================
# Creating First processing file after cleansing null values and get insights 
# of data from the html file
# =============================================================================

credit_df_profile = pandas_profiling.ProfileReport(credit_df_clean)
credit_df_profile.to_file(outputfile="Finacial_Credit_PostProcess_First.html")

# =============================================================================
# Describes the now 48 numerical columns their mean, std, and details.
# =============================================================================
credit_df_clean.describe()

# =============================================================================
# Need to encode 8 categorical values - from the post process html file
# Removing those from orifinal df and then after their encdong again mergig to 
# original one
# =============================================================================

credit_df_cat=credit_df_clean[['CODE_GENDER','FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_CONTRACT_TYPE',
                         'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE','NAME_INCOME_TYPE','WEEKDAY_APPR_PROCESS_START']]
credit_df_cat.shape

# =============================================================================
# =============================================================================
#credit_df_cat['NAME_FAMILY_STATUS'].unique()

dummy = pd.get_dummies(credit_df_cat, prefix=['CODE_GENDER','FLAG_OWN_CAR', 'FLAG_OWN_REALTY','NAME_CONTRACT_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','NAME_INCOME_TYPE','WEEKDAY_APPR_PROCESS_START'], 
                       columns=['CODE_GENDER','FLAG_OWN_CAR', 'FLAG_OWN_REALTY','NAME_CONTRACT_TYPE','NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE','NAME_INCOME_TYPE','WEEKDAY_APPR_PROCESS_START'])
credit_df_cat = credit_df_cat.drop(['CODE_GENDER','FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_CONTRACT_TYPE',
                         'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE','NAME_INCOME_TYPE','WEEKDAY_APPR_PROCESS_START'],axis = 1)
credit_df_cat = credit_df_cat.join(dummy)

credit_df_cat.columns

# =============================================================================
# Removing one column of each encoded data as this field is completely 
# unnecessary for futher processing.
# =============================================================================

credit_df_cat = credit_df_cat.drop(['CODE_GENDER_F','FLAG_OWN_CAR_N','FLAG_OWN_REALTY_N','NAME_CONTRACT_TYPE_Cash loans',
 'NAME_FAMILY_STATUS_Civil marriage','NAME_HOUSING_TYPE_Co-op apartment',
 'NAME_HOUSING_TYPE_House / apartment','NAME_INCOME_TYPE_Businessman',
 'WEEKDAY_APPR_PROCESS_START_FRIDAY'],axis = 1)

# =============================================================================
# Removing those columns from the original dataframe and the merge the new 
# encode categorical dataframe for futher processing.
# =============================================================================

credit_df_clean = credit_df_clean.drop(['CODE_GENDER','FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_CONTRACT_TYPE',
                         'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE','NAME_INCOME_TYPE','WEEKDAY_APPR_PROCESS_START'],
                         axis = 1)

final_credit_df = pd.concat([credit_df_clean, credit_df_cat], axis = 1)

# =============================================================================
# again creating profiling to get skew ness of data
#==============================================================================

credit_df_profile = pandas_profiling.ProfileReport(final_credit_df)
credit_df_profile.to_file(outputfile="Finacial_Credit_PostProcess_Second.html")

# =============================================================================
# from thre above html its clear that AMT_INCOME_TOTAL is skewed, so removing 
# the skewness using log function
#==============================================================================

temp_df = pd.DataFrame(final_credit_df['AMT_INCOME_TOTAL'])

df_log = np.log(temp_df.AMT_INCOME_TOTAL)
df_log.describe()

final_credit_df = final_credit_df.drop('AMT_INCOME_TOTAL',axis=1)

final_credit_df = final_credit_df.join(df_log)

# =============================================================================
# Now separting TARGET column from the DF to get y and also drop the same from 
# DF to get X 
# =============================================================================

y = final_credit_df['TARGET']

X = final_credit_df.drop(['NAME_INCOME_TYPE_Pensioner','TARGET','SK_ID_CURR'],axis=1)

# =============================================================================
# Now scale the data using MinMaxScaler  
# =============================================================================

from sklearn import preprocessing
saved_cols = X.columns
x = X.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X = pd.DataFrame(x_scaled, columns=saved_cols)

# =============================================================================
# Splitting the data for train and test set for model
# =============================================================================

from sklearn.model_selection import train_test_split
def split_data():
    return train_test_split(X, y, test_size=0.20, random_state=1) 
X_train, X_test, y_train, y_test = split_data()

# =============================================================================
# Model Selection using GridsearchCV 
# =============================================================================
import xgboost as xgb
from scipy import stats
from sklearn import model_selection
from scipy.stats import randint as sp_randint
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
X_tr, X_cv, y_tr, y_cv = model_selection.train_test_split(X_train, y_train, test_size=0.25)

# =============================================================================
# c_param={'learning_rate' :np.array([0.001,0.01,0.1]),
#   'n_estimators':sp_randint(100,1000),
#   'max_depth':sp_randint(1,10),
#   'min_child_weight':sp_randint(1,8),
#   'gamma':stats.uniform(0,0.02),
#   'subsample':stats.uniform(0.6,0.4),
#   'reg_alpha':sp_randint(0,200),
#   'reg_lambda':stats.uniform(0,200),
#   'colsample_bytree':stats.uniform(0.6,0.3)}
# =============================================================================
tuned_parameters = [{'learning_rate' :[0.1],'max_depth': [1, 5, 10, 50, 100],'n_estimators': [10,50,100,150,200]}]
model = GridSearchCV(XGBClassifier(), tuned_parameters, scoring = 'roc_auc', cv=2)

model.fit(X_tr, y_tr)

# =============================================================================
# Logistic Regression starts
# =============================================================================
print(model.best_estimator_)
print(model.score(X_cv, y_cv))

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)

y_pred_train = logreg.predict(X_train)  
y_pred_test = logreg.predict(X_test) 

from sklearn.metrics import accuracy_score
print('Accuracy score for test data is:', accuracy_score(y_test,y_pred_test))

from sklearn.metrics import confusion_matrix

confusion_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred_test))

print(confusion_matrix)

confusion_matrix.index = ['Actual Loan','Actual Not Loan']
confusion_matrix.columns = ['Predicted Loan','Predicted Not loan']
print(confusion_matrix)

feature_names = np.array(X.columns)
featureDict = dict(zip(feature_names, logreg))
sortedFeatures = sorted(featureDict.items(), key=lambda x: x[1],reverse=True)
print(type(sortedFeatures[0]))
features = []
for i in range(0,len(X.columns)-1):
    features.append(sortedFeatures[i][0])
    print(sortedFeatures[i])

X = X.drop(['FLAG_DOCUMENT_2','FLAG_DOCUMENT_4','FLAG_DOCUMENT_7',
            'FLAG_DOCUMENT_10','FLAG_DOCUMENT_12','FLAG_DOCUMENT_17',
            'FLAG_DOCUMENT_19','CODE_GENDER_XNA','NAME_FAMILY_STATUS_Unknown',
            'NAME_INCOME_TYPE_Maternity leave','NAME_INCOME_TYPE_Student'],axis=1)
# =============================================================================
# Logistic regression ends
# =============================================================================

import plotly.offline as offline
import plotly.graph_objs as go

def check_trade_off_xg(X_train,X_test,y_train,y_test):
    
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    
    [{'max_depth': [1, 5, 10, 50, 100]}]
    [{'n_estimators': [10,50,100,150,200]}]
    
    depth_range = [1, 5, 10, 50, 100]
    estim_range = [10,50,100,150,200]

    auc_scores =[]
    auc_train_scores = []

    i = 0
    for f, b in zip(depth_range, estim_range):
        clf =XGBClassifier(max_depth=f,n_estimators=b)

        # fitting the model on crossvalidation train
        clf.fit(X_train, y_train)

        
        #evaluate AUC score.
        probs = clf.predict_proba(X_test)
        probs = probs[:, 1]
        # calculate AUC
        auc = roc_auc_score(y_test, probs)
        print('AUC: %.3f' % auc)
        auc_scores.append(auc)
   
    print('#######################################################')
    print('AUC from train data ###################################')
    i = 0
    for f, b in zip(depth_range, estim_range):
        clf =XGBClassifier(max_depth=f,n_estimators=b)

        # fitting the model on crossvalidation train
        clf.fit(X_train, y_train)
        
        #evaluate AUC score.
        probs = clf.predict_proba(X_train)
        probs = probs[:, 1]
        # calculate AUC
        auc = roc_auc_score(y_train, probs)
        print('AUC: %.3f' % auc)
        auc_train_scores.append(auc) 

    trace1 = go.Scatter3d(x=depth_range,y=estim_range,z=auc_train_scores, name = 'train')
    trace2 = go.Scatter3d(x=depth_range,y=estim_range,z=auc_scores, name = 'Cross validation')
    data = [trace1, trace2]
    layout = go.Layout(scene = dict(
        xaxis = dict(title='n_estimators'),
        yaxis = dict(title='max_depth'),
        zaxis = dict(title='AUC'),))

    fig = go.Figure(data=data, layout=layout)
    offline.iplot(fig, filename='3d-scatter-colorscale')
    
check_trade_off_xg(X_tr,X_cv,y_tr,y_cv)


gbm = xgb.XGBClassifier(max_depth=5, n_estimators=150, learning_rate=0.1).fit(X_train, y_train)
predictions = gbm.predict(X_test)

def showHeatMap(con_mat):
    class_label = ["negative", "positive"]
    df_cm = pd.DataFrame(con_mat, index = class_label, columns = class_label)
    sns.heatmap(df_cm, annot = True, fmt = "d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

from sklearn.metrics import confusion_matrix
pred = gbm.predict(X_test)
con_mat = confusion_matrix(y_test, pred, [0, 1])

showHeatMap(con_mat)

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
acc = accuracy_score(y_test, pred) * 100
print('\nThe accuracy of the RF classifier for %f%%' % (acc))
probs = gbm.predict_proba(X_test)
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
      
# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.',label='test')
#plt.plot(fpr1, tpr1, marker='*',label='train')
plt.legend()
# show the plot
plt.show()



feature_names = np.array(X.columns)
featureDict = dict(zip(feature_names, gbm.feature_importances_))
sortedFeatures = sorted(featureDict.items(), key=lambda x: x[1],reverse=True)
print(type(sortedFeatures[0]))
features = []
for i in range(0,len(X.columns)-1):
    features.append(sortedFeatures[i][0])
    print(sortedFeatures[i])

X = X.drop(['FLAG_DOCUMENT_2','FLAG_DOCUMENT_4','FLAG_DOCUMENT_7',
            'FLAG_DOCUMENT_10','FLAG_DOCUMENT_12','FLAG_DOCUMENT_17',
            'FLAG_DOCUMENT_19','CODE_GENDER_XNA','NAME_FAMILY_STATUS_Unknown',
            'NAME_INCOME_TYPE_Maternity leave','NAME_INCOME_TYPE_Student'],axis=1)


X_train, X_test, y_train, y_test = split_data()
gbm = xgb.XGBClassifier(max_depth=5, n_estimators=150, learning_rate=0.1).fit(X_train, y_train)
predictions = gbm.predict(X_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
acc = accuracy_score(y_test, pred) * 100
print('\nThe accuracy of the RF classifier for %f%%' % (acc))
probs = gbm.predict_proba(X_test)
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
      
# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.',label='test')
#plt.plot(fpr1, tpr1, marker='*',label='train')
plt.legend()
# show the plot
plt.show()

import pickle as pkl

pkl.dump(gbm,open('LoanPrediction.pkl','wb'))

























