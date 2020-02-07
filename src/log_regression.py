import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import datetime
from collections import Counter

import statsmodels.api as sm
# from statsmodels.discrete.discrete_model import Logit
# from statsmodels.discrete.discrete_model import LogitResults
from sklearn.model_selection import train_test_split # X, y, random_state=9
from sklearn.preprocessing import StandardScaler # .fit(X_train), .transform(X) & y

from sklearn.metrics import accuracy_score # (y_test, y_pred)
from sklearn.metrics import classification_report # (y_test, y_pred)
from sklearn.metrics import confusion_matrix # (y_test, y_pred)

from pickle_process import load_files

def sort_split(df):
    """ return holdout, crossval, tank
    data randomly split among groups for modelling
    2x upsample 11449 purchase rows
    380127 rows main dfmodel
    """
    intforrand = 9
    purmask = df['made_purchase'] == 1
    pur = df[purmask]
    nopur = df[~purmask]
    pur = pur.sample(frac=3., replace=True, random_state=intforrand) # upsample to 34347
    upsamp_rows_size = pur.shape[0]
    # split pur
    holdpur = pur.sample(frac=0.25, random_state=intforrand) #df
    holdmask = pur.index.isin(holdpur.index)
    crosspur = pur[~holdmask] #df
    # split nopur
    times3_size = upsamp_rows_size * 3
    nopur_formodel = nopur.sample(n=times3_size, random_state=intforrand) # 103041
    holdnopur = nopur_formodel.sample(frac=0.25, random_state=intforrand) #df
    holdmaskno = nopur_formodel.index.isin(holdpur.index)
    crossnopur = nopur_formodel[~holdmaskno]  #df
    # set aside tank data unused for model
    tankmask = nopur.index.isin(nopur_formodel.index)
    dftank = nopur[~tankmask]  #df
    # combine for crossval and holdout
    holdappended = holdnopur.append(holdpur)
    crossappended = crossnopur.append(crosspur)
    holdout = holdappended.sample(frac=1., random_state=intforrand)
    crossval = crossappended.sample(frac=1., random_state=intforrand)
    return holdout, crossval, dftank

col_all = ['product_count', 'addtocart', 'view', 'time_hour']

def df_to_xy_constant(df):
    pass

def df_to_xy_standardized(df):
    X = df[col_all].to_numpy()
    y = df['made_purchase'].to_numpy().reshape(-1,1)
    scaler = StandardScaler().fit(X)
    Xstan = scaler.transform(X)
    return Xstan, y

def run_log_train(X, y, thresh=0.5):
    """ Logistic Regression on input data
    outputs logit_results for predictions on test data
    """
    logit = sm.Logit(y, X) # endog, exog
    logit_res = logit.fit()
    print(logit_res.summary2())
    print('exp(Coeffs)=Odds : ', np.exp(logit_res.params))
    # predict probabilities
    y_prob = logit_res.predict(X)
    y_pred = np.uint8(y_prob.reshape(-1,1) > thresh)
    true_count = Counter(y[:, 0])
    pred_count = Counter(y_pred[:, 0])
    print("y_train count: ", true_count, "y_pred count: ", pred_count)
    print("predict purchase right column\nactual purchase bottow row")
    print(confusion_matrix(y, y_pred))
    print("accuracy ", round(accuracy_score(y, y_pred), 3))
    return logit_res

def run_log_test(X, y, logit_results, thresh=0.5):
    """ uses trained model on test data
    """
    y_prob = logit_results.predict(X)
    y_pred = np.uint8(y_prob.reshape(-1,1) > thresh)
    true_count = Counter(y[:, 0])
    pred_count = Counter(y_pred[:, 0])
    print("y_test count: ", true_count, "y_pred count: ", pred_count)
    print("predict purchase right column\nactual purchase bottow row")
    print(confusion_matrix(y, y_pred))
    print("accuracy ", round(accuracy_score(y, y_pred), 3))


if __name__ == '__main__':

    dfmodel = pd.read_pickle('../../data/ecommerce/dfmodel_script.pkl', compression='zip')
    dfevents, dfcluster = load_files()
    print("dfmodel and dfcluster loaded")
    print("start df sort split function")
    holdout, crossval, tank = sort_split(dfmodel)
    Xcross, ycross = df_to_xy_standardized(crossval)
    Xhold, yhold = df_to_xy_standardized(holdout)
    print("run log model")
    trained_model = run_log_train(Xcross, ycross)
    run_log_test(Xhold, yhold, trained_model)



    
    
