import numpy as np
import pandas as pd

from utils import get_text_data, split_train_test, scale_features



def get_spectrum_features(dfs, k):
    # get all possible motifs
    motifs = set()
    
    for df in dfs:
        for code in df['seq']:
            n = len(code)
            for i in range(n - k + 1):
                motifs.add(code[i:i + k])
    
    ret_dfs = []
    
    for df in dfs:
        df_data = []
        
        for code in df['seq']:
            motif_count = {}
            n = len(code)
            for i in range(n - k + 1):
                motif = code[i:i + k]
                motif_count[motif] = motif_count.get(motif, 0) + 1
                
            df_data.append(motif_count)
        
        motif_data = {}
        for motif in motifs:
            motif_col = []
            for motif_count in df_data:
                motif_col.append(motif_count.get(motif, 0))
            
            motif_data[motif] = motif_col
        
        ret_dfs.append(pd.DataFrame.from_dict(motif_data))
    
    return motifs, ret_dfs
        

def get_nonconstant_features(df1, df2):
    columns1 = set(df1.columns[df1.nunique() != 0])
    columns2 = set(df2.columns[df1.nunique() != 0])
    return list(columns1 & columns2)


def prepare_kspectrum_data(dataset, path, k, scale_method, with_intercept, num_train):
    df_tr, df_y, df_te = get_text_data(dataset=dataset, path=path)
    _, (df_tr, df_te) = get_spectrum_features([df_tr, df_te], k)

    feature_cols = get_nonconstant_features(df_tr, df_te)
    df_tr, df_te = df_tr[feature_cols], df_te[feature_cols]

    X, y, X_te = df_tr.values, df_y.values, df_te.values

    if with_intercept:
        X = np.concatenate([np.ones([X.shape[0], 1]), X], axis=1)
        X_te = np.concatenate([np.ones([X_te.shape[0], 1]), X_te], axis=1)
    
    if scale_method is not None:
        X = scale_features(X, scale_method=scale_method)
        X_te = scale_features(X_te, scale_method=scale_method)

    X_train, y_train, X_val, y_val = split_train_test(X, y, num_train=num_train)

    return X_train, y_train, X_val, y_val, X_te

