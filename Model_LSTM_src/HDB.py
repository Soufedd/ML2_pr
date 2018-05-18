import numpy as np
import pandas as pd
import sys
from LearningClasses import GPRegressor, StackingAveragedModels, EnsembleRegressor, PartScaler
from MapeTimeCrossVal import MAPE_score, df_time_val_split, time_mape_cv, mape_cv, MAE_score
from Submission import part_sub, write_file
import pickle


from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.base import clone


import warnings

import xgboost as xgb
import lightgbm as lgb
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense

from GPy.kern import RBF, Brownian, Linear, MLP, Poly, PeriodicExponential as PE, Matern52

def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore warnings 

#Import Prepared Train and Test Dataset
with open(r"df_hdb_train.p", "rb") as input_file:
        df_train = pickle.load(input_file)

with open(r"df_hdb_test.p", "rb") as input_file:
        df_test = pickle.load(input_file)


pred_mean = pd.read_csv('pred_mean_hdb.csv').drop(columns = 'Unnamed: 0')

df_test = df_test.merge(pred_mean, left_on = ['monthtx'], right_on = ['monthtx'])

# cont_columns = np.arange(4)
# cat_columns = np.arange(4,12)



print('Training columns: {}'.format(list(df_train.columns)))

Towns = list(np.sort(list(set(df_test.town.values))))

#Init Models with scalers if needed
lasso = make_pipeline(PartScaler(last_col = 5), Lasso(alpha =0.1, random_state=1))
ENet = make_pipeline(PartScaler(last_col = 5), ElasticNet(alpha=0.0001, l1_ratio=.5, random_state=3))

gpr = GPRegressor(verbose = False, kernel = Brownian, normalizer = False, max_sample = 300)

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1,
                                   max_depth=10, max_features='auto',
                                   min_samples_leaf=10, min_samples_split=3, 
                                   loss='huber', random_state =5)

model_xgb = xgb.XGBRegressor(gamma=0.001, 
                             learning_rate=0.1, max_depth=30, 
                             min_child_weight=0.1, n_estimators=200,
                             reg_alpha=0.3, reg_lambda=0.4,
                              silent=1,
                             random_state =7, nthread = -1)
model_lgb = lgb.LGBMRegressor(n_estimators = 150, learning_rate = 0.15, random_state = 2,
                              boosting_type = 'gbdt', num_leaves = 90, subsample = 1,
                              reg_alpha = 0)
                              # learning_rate=0.8, n_estimators=200,
                              # max_bin = 55, bagging_fraction = 0.8,
                              # bagging_freq = 5, feature_fraction = 0.5,
                              # feature_fraction_seed=9, bagging_seed=9,
                              # min_data_in_leaf = 5, min_sum_hessian_in_leaf = 11, verbose = -1,
                              # min_data = 1, min_data_in_bin = 1, silent = 1)

model_slgb = lgb.LGBMRegressor(num_leaves=100, n_estimators = 600, learning_rate = 0.1, random_state = 2)#,
                                #boosting_type = 'rf')
model_sxgb = xgb.XGBRegressor(gamma=0.01, 
                             learning_rate=0.1, max_depth=7, 
                             min_child_weight=0.1, n_estimators=500,
                             reg_alpha=0.3, reg_lambda=0.4,
                              silent=1,
                             random_state =7, nthread = -1)


def NeuralNetModel():
  # create model
  model = Sequential()
  model.add(Dense(500, input_dim=4, kernel_initializer='normal', activation='relu'))
  model.add(Dense(1))
  # Compile model
  model.compile(loss='mean_absolute_percentage_error', optimizer='adam')
  return model

dnn = make_pipeline(PartScaler(last_col = 5), KerasRegressor(build_fn=NeuralNetModel, epochs=10, batch_size=200, verbose=0))

rfr = RandomForestRegressor(n_estimators = 10, n_jobs = -1, random_state = 123)

stacked_averaged_models = StackingAveragedModels(base_models = [model_slgb],
                                                 meta_model = model_sxgb,
                                                 n_folds = 4)



ensemble_reg = EnsembleRegressor([model_lgb])


#Mapes and Submission
mapes = []
tcv_mapes = []
cv_mapes = []
Sub = []
tcv = False
cv = False
val = False

#Cross-Val and Training
for Town in Towns:
  #Filter on the right town and keep the test index
  train_town = df_train.loc[df_train.town == Town]
  test_town = df_test.loc[df_test.town == Town]
  print("Filtered on "+ Town + '({}/{}) : {} rows in training dataset,\n {} rows in test dataset'.format(Towns.index(Town)+1,
  																									len(Towns),
  																									train_town.shape[0],
  																									test_town.shape[0]))





  test_ID = test_town['index']
  drop_cols_train = ['index', 'town', 'resale_price', 'mean_resale_price']#, 'monthtx']
  drop_cols_test = ['index', 'town']#, 'monthtx']

  test_town = test_town.drop(columns = drop_cols_test)
  train_town = train_town.drop(columns = drop_cols_train)

  #train_town = train_town.drop(columns = ['monthtx'])
  print(train_town.columns)



  #Get X_train, Y_train, X_test
  X_train = train_town.drop(columns = ['diff_mean']).values
  Y_train = train_town['diff_mean'].values.reshape(-1,1)
  X_test = test_town.values




  model = clone(model_lgb)

  #Time Cross Validation 
  if tcv:
    time_splits = df_time_val_split(train_town, n_splits = 3, s_ahead = 2)
    print('\nTime Cross Validation')
    score = time_mape_cv(model, X_train, Y_train, time_splits, MAPE_score, Y_transform_fn = np.log1p, Y_inv_transform_fn = np.expm1,
          verbose = True)
    print("\nTCV Model score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    tcv_mapes.append(score)

  #Regular Cross Validation 

  if cv:
    print('\nRegular Cross Validation')
    score = mape_cv(model, X_train, Y_train,Y_transform_fn = np.log1p, Y_inv_transform_fn = np.expm1, verbose = 2)
    print("\nCV Model score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    cv_mapes.append(score)

  #Training and Prediction
  if val:
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.05, random_state=42, shuffle = False)
  else:
    X_val = X_train
    Y_val = Y_train

  constant = - 2*np.min(Y_train) +1

  #Y Scaled
  # std_scaler = StandardScaler()
  # std_scaler.fit(Y_train)
  # Y_train = std_scaler.transform(Y_train)

  Y_train_scaled = np.log1p(Y_train + constant)
  Y_val_scaled = Y_val

  # Y_train_scaled = Y_train
  # Y_val_scaled = Y_val



  print('Training Models')
  model.fit(X_train, Y_train_scaled)
  model_val_pred = np.expm1(model.predict(X_val))-constant
  model_pred = np.expm1(model.predict(X_test[:,:-1]))-constant+X_test[:,-1]
  score = MAE_score(Y_val_scaled, model_val_pred)
  print('\n V Model Score: {:.4f} ({:.4f})\n'.format(score.mean(), score.std()))


  Sub.append(part_sub(test_ID, model_pred))
  mapes.append(score)


if not (val and tcv and cv):
  write_file(Sub, 'hdb', date = False)
mapes = np.array(mapes)
cv_mapes = np.array(cv_mapes)
tcv_mapes = np.array(tcv_mapes)
print(tcv_mapes)
print(cv_mapes)
print(mapes)
print('Model TCV MAE on training dataset: {} ({})'.format(tcv_mapes.mean(),tcv_mapes.std()))
print('Model CV MAE on training dataset: {} ({})'.format(cv_mapes.mean(),cv_mapes.std()))
print('MAE on validation dataset: {} ({})'.format(mapes.mean(),mapes.std()))









