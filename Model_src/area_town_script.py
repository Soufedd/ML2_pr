import pandas as pd 
import numpy as np
import itertools as it

# private = pd.read_csv('Long_lat_private.csv')
# hdb = pd.read_csv('Long_lat_hdb.csv')
print('area_town_script')

private_df = pd.read_csv('../Input/private_train.csv')
hdb_df = pd.read_csv('../Input/hdb_train.csv')


private_long = pd.DataFrame(private_df.groupby('area')['longitude'].mean()).reset_index()
private_lat = pd.DataFrame(private_df.groupby('area')['latitude'].mean()).reset_index()
private = private_long.merge(private_lat, left_on = 'area', right_on= 'area')

hdb_long = pd.DataFrame(hdb_df.groupby('town')['longitude'].mean()).reset_index()
hdb_lat = pd.DataFrame(hdb_df.groupby('town')['latitude'].mean()).reset_index()
hdb = hdb_long.merge(hdb_lat, left_on = 'town', right_on= 'town')




def dist(arr):
	return (arr**2).sum()

private_area = []
for val in private.values:
	private_area.append([val[0], val[1:3]])


hdb_town = []
for val in hdb.values:
	hdb_town.append([val[0], val[1:3]])

dist_town = []
for a,b in it.product(private_area, hdb_town):
	dist_town.append([a[0],b[0], dist(a[1]-b[1])])


dist_town = np.array(dist_town)


#dist_town = dist_town[dist_town[:,2].argsort()]
corresp = []
for area in private.values:
	corresp.append([area[0]])


for i in range(len(corresp)):
	mini = 10000000
	for j in range(len(dist_town)):
		if corresp[i][0] == dist_town[j,0]:
			if float(dist_town[j,2]) < mini:
				mini = float(dist_town[j,2])
				nearest = dist_town[j,1]
	corresp[i].append(nearest)

corresp[3][1] = 'BUKIT BATOK'



	

pd.DataFrame(corresp, columns = ['area', 'town']).to_csv('area_town.csv')
