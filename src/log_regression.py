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
    pur = pur.sample(frac=2., replace=True, random_state=intforrand) # upsample to 22898
    upsamp_rows_size = pur.shape[0]
    # split pur
    holdpur = pur.sample(frac=0.25, random_state=intforrand) #df
    holdmask = pur.index.isin(holdpur.index)
    crosspur = pur[~holdmask] #df
    # split nopur
    times4_size = upsamp_rows_size * 4
    nopur_formodel = nopur.sample(n=times4_size, random_state=intforrand) #  91592
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

def df_to_xy(df):
    pass

def dt_to_xy_standardized(df):
    pass


if __name__ == '__main__':

    dfmodel = pd.read_pickle('../../data/ecommerce/dfmodel_script.pkl', compression='zip')
    dfevents, dfcluster = load_files()
    print("dfmodel and dfcluster loaded")
    col_all = ['product_count', 'addtocart', 'view', 'time_hour']

    print("start df sort split function")
    holdout, crossval, tank = sort_split(dfmodel)

    
    
