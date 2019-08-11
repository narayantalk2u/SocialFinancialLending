# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 13:12:46 2019

@author: nmaharan
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')

import pickle as pkl

credit_df_test = pd.read_csv(r"D:\SapientFirstRound\Complete-Data-Set\application_test.csv")

credit_df_test_clean = credit_df_test.drop(['CNT_FAM_MEMBERS','DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE',
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

credit_df_test_clean['OWN_CAR_AGE'].fillna(0, inplace=True)
credit_df_test_clean['AMT_ANNUITY'].fillna(0, inplace=True)
credit_df_test_clean['APARTMENTS_AVG'].fillna(0, inplace=True)
credit_df_test_clean['EXT_SOURCE_2'].fillna(0, inplace = True)

credit_df_test_cat=credit_df_test_clean[['CODE_GENDER','FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_CONTRACT_TYPE',
                         'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE','NAME_INCOME_TYPE','WEEKDAY_APPR_PROCESS_START']]
credit_df_test_cat.shape



dummy = pd.get_dummies(credit_df_test_cat, prefix=['CODE_GENDER','FLAG_OWN_CAR', 'FLAG_OWN_REALTY','NAME_CONTRACT_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','NAME_INCOME_TYPE','WEEKDAY_APPR_PROCESS_START'], 
                       columns=['CODE_GENDER','FLAG_OWN_CAR', 'FLAG_OWN_REALTY','NAME_CONTRACT_TYPE','NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE','NAME_INCOME_TYPE','WEEKDAY_APPR_PROCESS_START'])
credit_df_test_cat = credit_df_test_cat.drop(['CODE_GENDER','FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_CONTRACT_TYPE',
                         'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE','NAME_INCOME_TYPE','WEEKDAY_APPR_PROCESS_START'],axis = 1)
credit_df_test_cat = credit_df_test_cat.join(dummy)

credit_df_test_cat.columns

credit_df_test_cat = credit_df_test_cat.drop(['CODE_GENDER_F','FLAG_OWN_CAR_N','FLAG_OWN_REALTY_N','NAME_CONTRACT_TYPE_Cash loans',
 'NAME_FAMILY_STATUS_Civil marriage','NAME_HOUSING_TYPE_Co-op apartment',
 'NAME_HOUSING_TYPE_House / apartment','NAME_INCOME_TYPE_Businessman',
 'WEEKDAY_APPR_PROCESS_START_FRIDAY'],axis = 1)

credit_df_test_clean = credit_df_test_clean.drop(['CODE_GENDER','FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_CONTRACT_TYPE',
                         'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE','NAME_INCOME_TYPE','WEEKDAY_APPR_PROCESS_START'],
                         axis = 1)

final_credit_df_test = pd.concat([credit_df_test_clean, credit_df_test_cat], axis = 1)

temp_df = pd.DataFrame(final_credit_df_test['AMT_INCOME_TOTAL'])

df_log = np.log(temp_df.AMT_INCOME_TOTAL)
df_log.describe()

final_credit_df_test = final_credit_df_test.drop('AMT_INCOME_TOTAL',axis=1)

final_credit_df_test = final_credit_df_test.join(df_log)


X = final_credit_df_test.drop(['NAME_INCOME_TYPE_Pensioner','SK_ID_CURR'],axis=1)


X = X.drop(['FLAG_DOCUMENT_2','FLAG_DOCUMENT_4','FLAG_DOCUMENT_7',
            'FLAG_DOCUMENT_10','FLAG_DOCUMENT_12','FLAG_DOCUMENT_17',
            'FLAG_DOCUMENT_19','CODE_GENDER_XNA','NAME_FAMILY_STATUS_Unknown',
            'NAME_INCOME_TYPE_Maternity leave','NAME_INCOME_TYPE_Student'],axis=1)

model = pkl.load(open(r'LoanPrediction.pkl','rb'))

test_pred = model.predict(X)

id = credit_df_test_clean['SK_ID_CURR']
id_arr = id.as_matrix()
saved_df = pd.DataFrame({'SK_ID_CURR':id_arr,'TARGET':test_pred})

saved_df.to_csv("output.csv")





