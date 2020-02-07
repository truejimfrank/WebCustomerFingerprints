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
    return dfevents, dfcluster

def time_format(df):
    """formats the timestamp column inplace"""
    times=[]  # convert from unix time format
    for i in df['timestamp']:
        times.append(datetime.datetime.fromtimestamp(i//1000.0))
    df['timestamp']=times

def toy_dataframe(df):
    """makes trial dataframe with two users with 'transactions'"""
    mask1 = df['visitorid'] == 1155978
    mask2 = df['visitorid'] == 539
    # selector = mask1 | mask2
    return df.loc[mask1 | mask2]

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

def onehot_the_df(df):
    """ add columns addtocart, transaction, view in this order
    also adds copied timestamp column for time_delta comlumns later on
    """
    df = df.assign(mintimestamp=lambda x: x.timestamp)
    enc = OneHotEncoder(sparse=False, dtype=np.uint8)
    X = df['event'].values.reshape(-1, 1)
    enc.fit(X)
    X = enc.transform(X)
    dfonehot = pd.DataFrame(X, columns=enc.categories_[0])
    return pd.concat([df.reset_index(), dfonehot], axis=1)

def agg_func(row):
    """
    function for agging events onehot df
    columns: 9 after onehot (subtracting 1 for visitorid to index)
    """
    # c0, c1, c2, c3, c4, c5, c6, c7 = [0,0,0,0,0,0,0,0]
    c0 = row['index']
    c1 = np.max(row['timestamp']) # preparing for time_delta assign
    c2 = row['event']
    c3 = np.unique(row['itemid']).shape[0] # number of items acted on
    c4 = np.max(row['transactionid']) # pulls an id if there is one
    c45 = np.min(row['mintimestamp']) # preparing for time_delta assign
    c5 = np.sum(row['addtocart'])
    c6 = np.sum(row['transaction'])
    c7 = np.sum(row['view'])
    return c0, c1, c2, c3, c4, c45, c5, c6, c7

def add_3_more_columns(df):
    """ rename column {"itemid":"product_count"}
    adding time delta columns and made_purchase y target
    drop some columns for clarity
    """
    df.rename(columns={"itemid":"product_count"}, inplace=True)
    df = df.assign(time_delta=lambda x: x.timestamp - x.mintimestamp)
    df = df.assign(time_hour=lambda x: x.time_delta / np.timedelta64(1, 'h'))
    df = df.assign(made_purchase=lambda x: np.uint8(x.transaction > 0))
    droplist = ['index', 'event', 'mintimestamp']
    df.drop(columns=droplist, inplace=True)   
    return df

def save_pickle(df, filepath):
    df.to_pickle(filepath, compression='zip')
    print("pickel saved! to:" + filepath)


if __name__ == '__main__':

    dfevents, dfcluster = load_files()
    print("completed file load function")
    time_format(dfevents)
    print("timestamp column formatted")

# make toydf and run functions on it
    dftoy = toy_dataframe(dfevents)
    print("made toy dataframe")
    dftoy = onehot_the_df(dftoy)
    print("onehot columns added")
    print("starting main groupby aggregation")
    dftagg = dftoy.groupby('visitorid').agg(agg_func)
    print("finished the main aggregation")
    dftagg = add_3_more_columns(dftagg)
    print("final columns added & dropped")
    toy_pickle_filepath = '../../data/ecommerce/dftoy_script.pkl'
    save_pickle(dftagg, toy_pickle_filepath)

# make the real dataframe and pickle save it (takes 6 minutes)
    print("starting the real dataframe functions")
    dfevents = onehot_the_df(dfevents)
    print("onehot columns added")
    print("starting main groupby aggregation")
    dfgroup = dfevents.groupby('visitorid').agg(agg_func)
    print("finished the main aggregation")
    dfgroup = add_3_more_columns(dfgroup)
    print("final columns added & dropped")
    real_pickle_filepath = '../../data/ecommerce/dfmodel_script.pkl'
    save_pickle(dfgroup, real_pickle_filepath)



