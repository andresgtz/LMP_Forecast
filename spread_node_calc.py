#Define high congestion, average congestion and low congestion nodes in the PML listing.
import pandas as pd
import os
from multiprocessing import Pool

#Create dictionary and insert each node and list of all prices
frames = []
nodeAvgSpread = {}
pool = Pool()   

def historicSpread(node):
	# Split dataframes into each node
	mask = global_data['Clave del nodo'] == node
	node_data = global_data[mask]
	node_data.set_index(['Fecha'], inplace=True,drop=False)
	a = node_data.groupby(level='Fecha')['Precio marginal local ($/MWh)'].min().copy()
	b = node_data.groupby(level='Fecha')['Precio marginal local ($/MWh)'].max().copy()
	spread = b.sub(a).to_frame('Spread Intradia')
	return spread['Spread Intradia'].mean()


### Read files and form global dataframe
# Iterate through all files and add node data to global dataframe
for filename in os.listdir('PML_DATA_original/'):
	print("Gathering info from: " + filename)
	URI = 'PML_DATA_original/'+filename
	temp = pd.read_csv(URI, date_parser=True)
	frames.append(temp)
global_data = pd.concat(frames, sort=False)

nodes = global_data['Clave del nodo'].unique()

for i in nodes:
	nodeAvgSpread[i] = historicSpread(i)

print(nodeAvgSpread)