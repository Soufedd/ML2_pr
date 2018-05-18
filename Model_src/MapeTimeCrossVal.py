import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_val_score




def MAPE_score(y_true,y_pred, Y_inv_transform_fn = None):
    if Y_inv_transform_fn is not None:
        y_pred = Y_inv_transform_fn(y_pred)
        y_true = Y_inv_transform_fn(y_true)
    y_true = y_true.reshape(1,-1)
    y_pred = y_pred.reshape(1,-1)
    return 100*np.abs((y_true-y_pred)/y_true).mean()


def df_time_val_split(train, n_splits, s_ahead = 2):
	time = train.monthtx.values
	#time = train
	min_time = np.min(time)
	time = time - min_time + 1
	max_time = np.max(time) - s_ahead
	train_val_time = []
	for n in range(n_splits+1):
		train_val_time.append([list(np.where(time <= n*(max_time//n_splits))[0]),
			list(np.where( (time > n*(max_time//n_splits)) & (time <= n*(max_time//n_splits) + s_ahead) )[0])])
	return train_val_time[1:]


def time_mape_cv(model, X , Y , time_splits, scorer, Y_transform_fn = None, Y_inv_transform_fn = None, verbose = False):
    scores = []
    if (Y_transform_fn is None and Y_inv_transform_fn is not None) or (Y_transform_fn is not None and Y_inv_transform_fn is None):
    	print('Issue with Y transform function')
    	return
    if Y_transform_fn is not None:
    	Y = Y_transform_fn(Y)
    j = 0
    for train_idx, val_idx in time_splits:
        if verbose:
            print('\nTCV step {}/{}'.format(j+1,len(time_splits)))
        X_train = X[train_idx]
        Y_train = Y[train_idx]
        X_val = X[val_idx]
        Y_val = Y[val_idx]
        model.fit(X_train,Y_train)
        Y_pred = model.predict(X_val).reshape(-1,1)
        if Y_inv_transform_fn is None:
            scores.append(scorer(Y_val, Y_pred))
        else:
            scores.append(scorer(Y_val, Y_pred, Y_inv_transform_fn))
        j+=1
    return(np.array(scores))

def mape_cv(model,X,Y, n_folds = 3, Y_transform_fn = None, Y_inv_transform_fn = None, verbose = 0):
    if (Y_transform_fn is None and Y_inv_transform_fn is not None) or (Y_transform_fn is not None and Y_inv_transform_fn is None):
        print('Issue with Y transform function')
        return
    if Y_transform_fn is not None:
        Y = Y_transform_fn(Y)
        MAPE = make_scorer(lambda y_true, y_pred: MAPE_score(y_true, y_pred, Y_inv_transform_fn = Y_inv_transform_fn)) 
    else:
        MAPE = make_scorer(MAPE_score)
    kfold = KFold(n_splits = n_folds, shuffle = True, random_state = 123).split(X)
    mape= cross_val_score(model, X, Y, scoring=MAPE, cv = kfold, verbose = verbose)
    return(mape)











