import pandas as pd
import numpy as np
from pprint import pprint
from pickle_process import load_files
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim.models import LdaMulticore


def run_gensim_lda(dfcluster, num_cluster=8):
    """ runs gensim LDA
    returns the trained LDA model & bow_corpus
     --- options for gensim LDA ---
    class gensim.models.ldamulticore.LdaMulticore(corpus=None, num_topics=100, 
    id2word=None, workers=None, chunksize=2000, passes=1, batch=False, alpha='symmetric', 
    eta=None, decay=0.5, offset=1.0, eval_every=10, iterations=50, 
    gamma_threshold=0.001, random_state=None, minimum_probability=0.01, 
    minimum_phi_value=0.01, per_word_topics=False, dtype=<class 'numpy.float32'>)
    """
    gensim_processed = dfcluster['product_list']
    # create dictionary
    id2word = gensim.corpora.Dictionary(gensim_processed)
    # Filter out tokens that appear in (tokens = products) less than 3 documents
    id2word.filter_extremes(no_below=3, no_above=0.5, keep_n=200000)
    print("number of features in lda model: ", len(id2word))
    bow_corpus = [id2word.doc2bow(text) for text in gensim_processed]
    print(f"number clusters option set at {num_cluster}")
    print("running LdaMulticore. this will take a minute or two")
    lda_model = LdaMulticore(bow_corpus, num_topics=num_cluster, id2word=id2word, 
                            passes=2, iterations=50, workers=3, random_state=9)
    return lda_model, bow_corpus
    
def print_lda(lda_model, bow_corpus):
    print('\nPerplexity: ', lda_model.log_perplexity(bow_corpus))  
    # a measure of how good the model is. lower the better.

def make_join_probs_df(dfmodel, lda_model, bow_corpus, num_cluster=8):
    """ make and join probs df
    """
    empty = pd.DataFrame(columns=range(num_cluster))
    for idx, vector in enumerate(bow_corpus):
        # pprint(lda_model.get_document_topics(vector))
        probslist = lda_model.get_document_topics(vector)
        npzero = np.zeros(num_cluster).reshape(1, num_cluster)
        dfrow = pd.DataFrame(npzero)
        for tup in probslist:
            # print(tup[0], tup[1])
            dfrow.iloc[0, tup[0]] = tup[1] # [0]col idx [1] prob
        empty = empty.append(dfrow)
        if idx % 50000 == 0:
            print(f"at row {idx}")
    return pd.concat([dfmodel.reset_index(), empty.reset_index()], axis=1)


if __name__ == '__main__':

    dfmodel = pd.read_pickle('../../data/ecommerce/dfmodel_script.pkl', compression='zip')
    dfevents, dfcluster = load_files() # dfcluster is 380K

    print("files are loaded. starting gensim lda")
    lda_model, bow_corpus = run_gensim_lda(dfcluster, num_cluster=8)
    print("LDA model complete\nstarting make join probs df")
    dfmodel = make_join_probs_df(dfmodel, lda_model, 
                                bow_corpus, num_cluster=8)
    print(dfmodel.head())
    print("fin")


