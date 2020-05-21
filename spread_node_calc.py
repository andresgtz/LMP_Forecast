from __future__ import division
import pandas as pd
import os
from multiprocessing import Pool, cpu_count, Manager
import sys
import tqdm

#Create dictionary and insert each node and list of all prices
frames = []
manager = Manager()
nodeAvgSpread = manager.dict()

def historicSpread(node):
	# Split dataframes into each node
	mask = global_data['Clave del nodo'] == node
	node_data = global_data[mask]
	node_data.set_index(['Fecha'], inplace=True,drop=False)
	a = node_data.groupby(level='Fecha')['Precio marginal local ($/MWh)'].min().copy()
	b = node_data.groupby(level='Fecha')['Precio marginal local ($/MWh)'].max().copy()
	spread = b.sub(a).to_frame('Spread Intradia')
	nodeAvgSpread[node] = spread['Spread Intradia'].mean()


### Read files and form global dataframe
# Iterate through all files and add node data to global dataframe
for filename in os.listdir('PML_DATA_original/'):
	print("Gathering info from: " + filename)
	URI = 'PML_DATA_original/'+filename
	temp = pd.read_csv(URI, date_parser=True)
	frames.append(temp)
global_data = pd.concat(frames, sort=False)

nodes = global_data['Clave del nodo'].unique()



#print(len(global_data.loc[global_data['Clave del nodo']=='04ECC-400'].groupby('Fecha')))

print("\nObtaining average intraday spreads for all nodes: ")
pool = Pool(processes=5)
#pool.map(historicSpread, nodes) 
for _ in tqdm.tqdm(pool.imap_unordered(historicSpread, nodes), total=len(nodes)):
    pass


#Sort dictionary to obtain high, med and low congestion nodes
print("Sorting dictionary")
s_nodeAvgSpread = sorted(nodeAvgSpread.items(), key=lambda x: x[1], reverse=True)    

print("Number of nodes: ", len(s_nodeAvgSpread))


print("Top 5")
for i in range(0, len(s_nodeAvgSpread)):
	print(s_nodeAvgSpread[0 + i],len(global_data.loc[global_data['Clave del nodo'] == s_nodeAvgSpread[0 + i][0]].groupby('Fecha')))


#print("Mid 5")
#for i in range(0,20):
# 	print(s_nodeAvgSpread[int(len(s_nodeAvgSpread)/2)+i],len(global_data.loc[global_data['Clave del nodo']==s_nodeAvgSpread[int(len(s_nodeAvgSpread)/2)+i][0]].groupby('Fecha')))


# print("Bot 5")
# for i in range(-100,0):
# 	print(s_nodeAvgSpread[int(len(s_nodeAvgSpread))+i],len(global_data.loc[global_data['Clave del nodo']==s_nodeAvgSpread[int(len(s_nodeAvgSpread))+i][0]].groupby('Fecha')))
