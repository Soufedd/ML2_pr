import numpy as np
import pandas as pd
import pickle
from pandas.api.types import CategoricalDtype as cattype


features_train = ['type_of_land', 'property_type','type_of_sale', 'price','area','floor_area_sqm','latitude','longitude','floor_num','monthtx']
features_test = ['index','type_of_land', 'property_type','type_of_sale','area','floor_area_sqm','latitude','longitude','floor_num','monthtx']


#Formating and Removing Outliers

print('Private Data Preprocessing')


df = pd.read_csv('../Input/private_train.csv')

property_type = list(set(df.property_type.values))
type_of_sale = list(set(df.type_of_sale.values))
type_of_land = list(set(df.type_of_land.values))

df['monthtx'] = df['month'].map(lambda x: int(x.split('-')[0])*12+int(x.split('-')[1]))

#df = df.loc[df.monthtx >= 24121]
df = df.loc[df.price <= 30000000]
df = df.loc[df.floor_area_sqm <= 7000]
df = df.loc[df.latitude < 10]


area_town_corresp = pd.read_csv("area_town.csv")

drop_first = True



df = df.filter(items = features_train)
df = df.reset_index(drop=True)
df = df.merge(area_town_corresp, left_on = 'area', right_on = 'area')
df = df.drop(columns = ['area', 'Unnamed: 0'])
df = df.rename(columns = {'price': 'resale_price', 'floor_num': 'floor'})

Towns = np.sort(list(set(df.town.values)))
value = {}
for town in Towns:
	value[town] = {'floor': df.loc[df.town == town].floor.median()}
	df.loc[df.town == town] = df.loc[df.town == town].fillna(value = value[town])
df = df.fillna(value = {'floor': df.floor.median()})

df['property_type'] = df['property_type'].astype(cattype(categories=property_type))
df['type_of_sale'] = df['type_of_sale'].astype(cattype(categories=type_of_sale))
df['type_of_land'] = df['type_of_land'].astype(cattype(categories=type_of_land))
df = pd.get_dummies(df,drop_first=drop_first, columns=['type_of_land', 'type_of_sale', 'property_type'])

pickle.dump(df, open("df_private_train.p", "wb"))


#Test

df = pd.read_csv("../Input/private_test.csv")
df['monthtx'] = df['month'].map(lambda x: int(x.split('-')[0])*12+int(x.split('-')[1]))
df = df.filter(items=features_test)
df = df.merge(area_town_corresp, left_on = 'area', right_on = 'area')
df = df.drop(columns = ['area', 'Unnamed: 0'])
df = df.rename(columns = {'floor_num': 'floor'})
df['property_type'] = df['property_type'].astype(cattype(categories=property_type))
df['type_of_sale'] = df['type_of_sale'].astype(cattype(categories=type_of_sale))
df['type_of_land'] = df['type_of_land'].astype(cattype(categories=type_of_land))
df = pd.get_dummies(df,drop_first=drop_first,columns=['type_of_land', 'type_of_sale', 'property_type'])

pickle.dump(df, open("df_private_test.p", "wb"))

