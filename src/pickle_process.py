import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import OneHotEncoder

"""
Data Pipeline
Load and process events.csv into the dataframe for LogisticRegression
"""

def load_files():
    dfevents = pd.read_csv('../../data/ecommerce/events.csv')
    print("events dataframe shape\t", dfevents.shape)
    dfcluster = pd.read_pickle('../../data/ecommerce/cust_prod_clust2.pkl', compression='zip')
    dfcluster = dfcluster[dfcluster['item_count'] < 400] # product count threshold
    seriesldamask = pd.read_pickle('../../data/ecommerce/seriesldamask.pkl', compression='zip')
    dfcluster = dfcluster[seriesldamask.values]
    print("products dataframe shape (n unique users)\t", dfcluster.shape)
    clusterlist = list(dfcluster.index.values)
    clustermask = dfevents['visitorid'].isin(clusterlist)
    dfevents = dfevents.loc[clustermask]
    print("events dataframe shape after product filtering\t", dfevents.shape)
    return dfevents

def time_format(df):
    """formats the timestamp column inplace"""
    times=[]  # convert from unix time format
    for i in df['timestamp']:
        times.append(datetime.datetime.fromtimestamp(i//1000.0))
    df['timestamp']=times

def event_counts(df):
    """
    OBSOLETE : DOESN'T WORK
    input events df
    returns 3 DF for view_count, add_count, purchase_count
    """
    col_map = [('view', 'view_count'), ('addtocart', 'add_count'), 
               ('transaction', 'purchase_count')]
    dft = [0, 0, 0]
    for idx, tup in enumerate(col_map):
        dft[idx] = df.loc[df['event'] == tup[0], ['visitorid', 
                                'event']].groupby('visitorid').agg('count')
    return dft[0], dft[1], dft[2]

def set_counts(df):
    """
    OBSOLETE : DOESN'T WORK
    input dfevents to set: view_count, add_count, purchase_count
    """
    df['view_count'] = 0
    df['add_count'] = 0
    df['purchase_count'] = 0
    dfview, dfadd, dfpurchase = event_counts(df)
    #     viewmask = df['visitorid'].isin(dfview.index.values)
    #     addmask = df['visitorid'].isin(dfadd.index.values)
    #     purchasemask = df['visitorid'].isin(dfpurchase.index.values)
    for visid in df.visitorid.values:
        if visid in dfview.index.values:
            df.loc[[df['visitorid'] == visid], ['view_count']] = dfview.loc[visid]
    #     for visid in df[['visitorid']].values:
        if visid in dfadd.index.values:
            df.loc[[df['visitorid'] == visid], ['add_count']] = dfadd.loc[visid]
    #     for visid in df[['visitorid']].values:
        if visid in dfpurchase.index.values:
            df.loc[[df['visitorid'] == visid], ['purchase_count']] = dfpurchase.loc[visid]

def another_counter(df):
    """
    OBSOLETE : DOESN'T WORK (didn't complete running in 8+ hours)
    input dfevents to set: view_count, add_count, purchase_count
    maybe this function will work
    """
    uniq_id = np.unique(df['visitorid'].to_numpy())
    for visid in uniq_id:
        id_mask = df['visitorid'] == visid
        dfslice = df.loc[id_mask]
        view = sum(dfslice['event'] == 'view')
        add = sum(dfslice['event'] == 'addtocart')
        purchase = sum(dfslice['event'] == 'transaction')
        df.loc[id_mask, ['view_count']] = view
        df.loc[id_mask, ['add_count']] = add
        df.loc[id_mask, ['purchase_count']] = purchase

def onehot_the_df():
    pass

def agg_func(row):
    """
    function for agging events onehot df
    columns: 8 after onehot (subtracting 1 for visitorid to index)
    """
    # c0, c1, c2, c3, c4, c5, c6, c7 = [0,0,0,0,0,0,0,0]
    c0 = row['index']
    c1 = row['timestamp']
    c2 = row['event']
    c3 = np.unique(row['itemid']).shape[0] # number of items acted on
    c4 = np.max(row['transactionid'])
    c5 = np.sum(row['addtocart'])
    c6 = np.sum(row['transaction'])
    c7 = np.sum(row['view'])
    return c0, c1, c2, c3, c4, c5, c6, c7

def add_3_more_columns():
    pass



if __name__ == '__main__':

    dfevents = load_files()
    print("completed file load function")
    time_format(dfevents)
    print("timestamp column formatted")


