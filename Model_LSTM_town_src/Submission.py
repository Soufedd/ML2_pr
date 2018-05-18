import pandas as pd
import datetime



def part_sub(index, part_result):
	sub = pd.DataFrame()
	sub['index'] = index
	sub['price'] = part_result
	return sub

def write_file(subs, hdb_private = '', date = True):
	sub = pd.concat(subs)
	sub.sort_values(by = ['index'],inplace = True)
	if date:
		out_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
	else: 
		out_time = ''
	sub.to_csv('../Output/submission_'+hdb_private+out_time+'.csv',index=False)

def concat_csv(csv1,csv2):
	write_file([pd.read_csv(csv1 + '.csv'),pd.read_csv(csv2 + '.csv')])
