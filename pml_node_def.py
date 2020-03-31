#Define high congestion, average congestion and low congestion nodes in the PML listing.
import pandas as pd
import os

#Create dictionary and insert each node and list of all prices
node_d = {}

#Iterate through all files and add node data to the dictionary
print("Creating dictionary")
for filename in os.listdir('PML_DATA/'):
	print("Gathering info from: " + filename)
	URI = 'PML_DATA/'+filename
	data = pd.read_csv(URI, date_parser=True)
	for row in data.iterrows():
		if row[1]['Clave del nodo'] not in node_d.keys():
			node_d[row[1]['Clave del nodo']] = 0
		else:
			node_d[row[1]['Clave del nodo']] += row[1]['Componente de congestion ($/MWh)']

#Sort dictionary to obtain high, med and low congestion nodes
print("Sorting dictionary")
s_node_d = sorted(node_d.items(), key=lambda x: x[1], reverse=True)    

#Print nodes
print("High congestion: ")
print(s_node_d[0])
print("Medium congestion: ")
print(s_node_d[int(len(s_node_d)/2)])
print("Low congestion: ")
print(s_node_d[len(s_node_d)-1])