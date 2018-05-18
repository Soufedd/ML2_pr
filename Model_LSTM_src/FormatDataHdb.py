import numpy as np
import pandas as pd
import pickle
from pandas.api.types import CategoricalDtype as cattype



features_train = ['index','monthtx','floor_area_sqm', 'floor', 'lease_commence_date', 'town', 'latitude', 'flat_model', 'resale_price']
features_test =  ['index','monthtx','floor_area_sqm', 'floor', 'lease_commence_date', 'town', 'latitude', 'flat_model']

print('HDB Data Preprocessing')


#train
df = pd.read_csv("../Input/hdb_train.csv")
flat_model = list(set(df.flat_model.values))
df['monthtx'] = df['month'].map(lambda x: int(x.split('-')[0])*12+int(x.split('-')[1]))

#df = df.loc[df.monthtx >= 24121]
df = df.loc[df.resale_price <= 30000000]
df = df.loc[df.floor_area_sqm <= 7000]
df = df.loc[df.latitude < 10]

df = df.filter(items = features_train)

df['flat_model'] = df['flat_model'].astype(cattype(categories=flat_model))

df = pd.get_dummies(df,drop_first=True, columns=['flat_model'])
df = df.reset_index(drop=True)

monthly_mean = df.groupby(['monthtx'])['resale_price'].mean()
monthly_mean = pd.DataFrame(monthly_mean).reset_index()
monthly_mean.columns = ['monthtx', 'mean_resale_price']

monthly_mean.to_csv("df_hdb_mean_time.csv")

df = df.merge(monthly_mean, left_on = ['monthtx'], right_on = ['monthtx'])

df['diff_mean'] = df.resale_price.values - df.mean_resale_price.values


pickle.dump(df, open("df_hdb_train.p", "wb"))


#test
df = pd.read_csv("../Input/hdb_test.csv")
df['monthtx'] = df['month'].map(lambda x: int(x.split('-')[0])*12+int(x.split('-')[1]))

df = df.filter(items = features_test)

df['flat_model'] = df['flat_model'].astype(cattype(categories=flat_model))

df = pd.get_dummies(df,drop_first=True, columns=['flat_model'])


pickle.dump(df, open("df_hdb_test.p", "wb"))




