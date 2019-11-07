#!/usr/bin/env python
# coding: utf-8

# we are getting there! so we first focus on breaking in front, any "in front", for each template sequence we extract a subgraph of possible nodes, then the subgraph of pre_post nodes (i dont think it matters distinguishing them) and then we further trim the graphs by taking only the valid paths seen in the data + extraction of the raw data pointers. another thing we could do is preparing a unique html page with that stuff!

# # TODO: 
# - these todo could also be fixed when we deal with cutin etc
# - review the old dictionary-template matching function from OLD_03_search_function_and_visualisations.ipynb
# - add possibilities of "*" values
# - figure out an elegant way to deal with int-dict-index and str-dict-index
# - merge the rendering of point nodes with image nodes

# # IMPORTS

# In[1]:


import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import re
from copy import deepcopy
import os
import sys


# In[2]:


from imports.general_settings import *
from imports.general_functions import *
from imports.NGram_KNrealRecursion import recursive_NGramKneserNey
from pyvis.network import Network, Node, Edge


	
class braking_in_front_subgraph_scenario_extraction(object) :
	
	def __init__(self, raw_input_filename, print_debug) :
		self.file_and_folder_preparation(raw_input_filename, print_debug)
		self.scenario_indices = ['n', 'p', 's']
		self.scenario_indices_long = ['nothing', 'pre_post', 'scenario']
		
	def file_and_folder_preparation(self, raw_input_filename, print_debug=False) :
		self.stem_filename = raw_input_filename[:raw_input_filename.find(".hdf5")]

		data_folder = '..\\..\\data\\'
		baseFolder = data_folder + projectName + '\\'

		self.inputFolder_dictionary = "01_activity_dictionary"
		
		inputFolder_data = "02_mapped_dataset"
		gridSizeSubfolder = str(len(valid_relative_positions['lateral'])-1) + "x" + str(len(valid_relative_positions['longitudinal'])-1)


		self.inputPath_data = baseFolder + inputFolder_data + "\\" + gridSizeSubfolder + "\\"


		self.dictionary_filename_stem = "activityDictionary"
		self.dataset_filename_stem = "mappedDataset"
		self.nGramGraph_filename_stem = "nGramGraph"
		self.networkX_filename_stem = "networkx"

		self.dataset_subnetwork_filename_stem = "dataset_subnetwork"
		self.dataset_subnetwork_img_filename_stem = "dataset_subnetwork_img"
		self.full_dataset_network_filename_stem = "full_dataset_network"
		self.scenario_subnetwork_img_filename_stem = "scenario_subnetwork_img"

		
		self.inputFolder_dictionaryImg = "img_" + gridSizeSubfolder + '\\'

		self.imgExtension = ".jpg"

		self.dictionary_filename = self.dictionary_filename_stem + "_" + gridSizeSubfolder + ".csv"

		self.inputPath_dictionary = baseFolder + self.inputFolder_dictionary + "\\"
		self.inputPath_dictionaryImg = self.inputPath_dictionary + self.inputFolder_dictionaryImg

		#inputPath_dictionaryImg

		output_folder = "03_retrieved_scenarios"

		self.output_path = baseFolder + output_folder + "\\" + self.stem_filename + "\\"

		if print_debug :
			print("** FILE AND FOLDER SETTINGS **")
			print("\tstem_filename:", self.stem_filename)
			print("\tbaseFolder:",baseFolder)
			print("\tinputPath_data:", self.inputPath_data)
			print("\tgridSizeSubfolder:",gridSizeSubfolder)
			print("\tinputFolder_dictionaryImg:",self.inputFolder_dictionaryImg)
			print("\tdictionary_filename:",self.dictionary_filename)
			print("\tinputPath_dictionary:",self.inputPath_dictionary)
			print("\tinputPath_dictionaryImg:",self.inputPath_dictionaryImg)
			print("\toutput_path:",self.output_path)
			#print("\t:",)
							
		if print_debug :
			print("\n** FOLDER INITIALISATIONS **")
			
		if output_folder not in os.listdir(baseFolder) :
			print("\tcreating", baseFolder + output_folder)
			os.mkdir(baseFolder + output_folder)
		else :
			print("\tfolder", output_folder, "already exists")

		if self.stem_filename not in os.listdir(baseFolder + output_folder) :
			print("\tcreating", self.stem_filename)
			os.mkdir(self.output_path)
		else :
			print("\tfolder", self.stem_filename, "already exists")
			
			
	def load_dictionary(self, print_debug) :
		
		if print_debug :
			print("\t01. loading dictionary", self.dictionary_filename, "..", end='')
	
		self.dictionary = pd.read_pickle(self.inputPath_dictionary + self.dictionary_filename)
		
		if print_debug :
			print("OK, shape =", self.dictionary.shape)
			
	def extract_breaking_in_front_nodes(self, print_debug) :
	
		if print_debug :
			print("\t02. extracting braking in front entries..")
	
		dict_valid_scenario_nodes = []
		#print(len(dictionary))
		for i in range(len(self.dictionary)) :
			if print_debug :
				print("\t\t", i, '/', len(self.dictionary), 6*' ', end='\r')
			entry = dict(self.dictionary.iloc[i])
			for t in range(8) :
				if entry['target_' + str(t) + '_relative_position_lateral'] == 'same_lane' :
					if entry['target_' + str(t) + '_relative_position_longitudinal'] in ['0<=x<=10', '10<x<=30'] :
						if entry['target_' + str(t) + '_longitudinal'] == 'd' :
							dict_valid_scenario_nodes.append(i)
							break
		dict_valid_scenario_nodes = list(set(dict_valid_scenario_nodes))
		
		dict_valid_scenario_nodes = [str(x) for x in dict_valid_scenario_nodes]
		
		if print_debug :
			print()
			print("\t\tfound", len(dict_valid_scenario_nodes), "nodes satisfying breaking in front")
	
		return dict_valid_scenario_nodes
		
	def load_dataset(self, file_to_parse, print_debug) :
	
		if print_debug :
			print("\t03. loading dataset..", end='')
	
		input_filename = self.dataset_filename_stem + "_" + self.stem_filename + ".csv"
		self.dataframe = pd.read_csv(self.inputPath_data + input_filename, dtype='str')
		
		if print_debug :
			print(self.dataframe.shape, end=' ')
			
		#nGramGraph_filename = self.nGramGraph_filename_stem + "_" + self.stem_filename + ".p"
		#ngram_model = pickle.load(open(inputPath_data + nGramGraph_filename, 'rb'))	
		
		networkX_filename = self.networkX_filename_stem + "_" + self.stem_filename + ".p"
		self.model_nx = pickle.load(open(self.inputPath_data + networkX_filename, "rb"))
		
		if print_debug :
			print(self.model_nx.number_of_nodes(), self.model_nx.number_of_edges(), "OK")
			
	def dataset_node_identification(self, dict_valid_scenario_nodes, print_debug) :
		# # DATASET NODE IDENTIFICATION
		# - we need to partition the dataset network into 3 node types:
		#	 - scenario nodes
		#	 - pre_post scenario nodes
		#	 - non relevant nodes
		# - scenario nodes are the nodes in the dataset which are also part of dict_valid_scenario_nodes
		# - pre_post are extracted accordingly - predecessors
		# 
		# - we also represent the information via network colouring
		
		if print_debug :
			print("\t03. dataset node identification..")
			
		if print_debug :
			print("\t\ta. find valid scenario nodes") # i.e. all nodes in dataset which are part of dict_valid_scenario_nodes")
			
		df_nodes = list(self.dataframe['dictionary_index'].drop_duplicates())
		df_scenario_nodes = sorted(list(set(df_nodes).intersection(dict_valid_scenario_nodes)))
	
		if print_debug :
			print("\t\t  the dataset has", len(df_scenario_nodes), "scenario nodes")
			
		if print_debug :
			print("\t\t2. extract the pre/post scenario nodes") # i.e. all predecessors/successors of scenario nodes which are not scenario nodes
			
		pre_nodes = []
		for node in df_scenario_nodes :
			pre_nodes += list(set(self.model_nx.predecessors(node)).difference(df_scenario_nodes))
		pre_nodes = list(set(pre_nodes))
		#len(pre_nodes)
		
		post_nodes = []
		for node in df_scenario_nodes :
			
			post_nodes += list(set(self.model_nx.successors(node)).difference(df_scenario_nodes))
		post_nodes = list(set(post_nodes))
		#len(post_nodes)
		
		df_pre_post_nodes = pre_nodes + post_nodes
		df_pre_post_nodes = list(set(df_pre_post_nodes))
		if print_debug :
			print("\t\t  the dataset has", len(df_pre_post_nodes), "pre-post scenario nodes")
			
		df_remaining_nodes = list(set(self.model_nx.nodes()).difference(df_scenario_nodes + df_pre_post_nodes))
		
		if print_debug :
			print("\t\t  remaining nodes:", len(df_remaining_nodes))

		#len(df_remaining_nodes) + len(df_pre_post_nodes) + len(df_scenario_nodes) == model_nx.number_of_nodes()
		
		return df_scenario_nodes, df_pre_post_nodes, df_remaining_nodes

	def generate_first_networks(self, print_debug) :
		if print_debug :
			print("\t04. full dataset network generation..", end='')
			
		full_dataset_network = Network(height=500, width=1200, directed=True, notebook=False)
		
		for node in self.df_remaining_nodes :
			full_dataset_network.add_node(int(node), node, color='black')
			
		for node in self.df_pre_post_nodes :
			full_dataset_network.add_node(int(node), node, color='blue')

		for node in self.df_scenario_nodes :
			full_dataset_network.add_node(int(node), node, color='green')

		for edge in list(self.model_nx.edges()) :
			full_dataset_network.add_edge(int(edge[0]), int(edge[1]))
			
		full_dataset_network.save_graph(self.full_dataset_network_filename_stem + ".html")
		
		if print_debug :
			print("OK")
			
		if print_debug :
			print("\t05. scenario subgraph extraction..", end='')
			
		scenario_subgraph_nx = self.model_nx.subgraph(self.df_pre_post_nodes + self.df_scenario_nodes)
		
		dataset_subnetwork = Network(height=500, width=1200, directed=True, notebook=False)

		for node in self.df_pre_post_nodes :
			dataset_subnetwork.add_node(int(node), node, color='blue')

		for node in self.df_scenario_nodes :
			dataset_subnetwork.add_node(int(node), node, color='green')

		for edge in list(self.model_nx.edges()) :
			if edge[0] in self.df_pre_post_nodes and edge[1] in self.df_scenario_nodes :
				dataset_subnetwork.add_edge(int(edge[0]), int(edge[1]))
			elif edge[0] in self.df_scenario_nodes and edge[1] in self.df_pre_post_nodes :
				dataset_subnetwork.add_edge(int(edge[0]), int(edge[1]))
			elif edge[0] in self.df_scenario_nodes and edge[1] in self.df_scenario_nodes :
				dataset_subnetwork.add_edge(int(edge[0]), int(edge[1]))

		for i in range(len(dataset_subnetwork.edges)) :
			dataset_subnetwork.edges[i]['color'] = edgeProperties['color']

		dataset_subnetwork.save_graph(self.dataset_subnetwork_filename_stem + ".html")
		
		if print_debug :
			print("OK")
			
		if print_debug :
			print("\t06. scenario subgraph with images extraction..", end='')
		
		dataset_subnetwork_img = deepcopy(dataset_subnetwork)

		for i in range(len(dataset_subnetwork_img.nodes)) :
			node_id = dataset_subnetwork_img.nodes[i]['id']
			dataset_subnetwork_img.nodes[i]['shape'] = 'image'
			dataset_subnetwork_img.nodes[i]['image'] = self.inputPath_dictionaryImg + str(node_id) + self.imgExtension
			
		for i in range(len(dataset_subnetwork_img.edges)) :
			dataset_subnetwork_img.edges[i]['color'] = edgeProperties['color']

		dataset_subnetwork_img.save_graph(self.dataset_subnetwork_img_filename_stem + ".html")
		
		if print_debug :
			print("OK")
			
		return full_dataset_network, dataset_subnetwork, dataset_subnetwork_img
		
	def assign_scenario_label(self, dict_index) :
		x = str(dict_index)
		if x in self.df_scenario_nodes :
			return self.scenario_indices.index("s")
		elif x in self.df_pre_post_nodes :
			return self.scenario_indices.index("p")
		else :
			return self.scenario_indices.index("n")
		
	def scenario_instance_extraction(self, print_debug) :
		# # SCENARIO INSTANCE EXTRACTION
		# it's a multilevel thingie:
		# - the dataset now should ahve labels as "pre_post", "scenario", "nothing"
		# - then we can shrink the dataset to have unique repetitions for the sequence
		# - in that way we can just extract sequencese "pre_post"-"scenario"-"pre-post"
		# - we will have to see who we can deal with multi sequence scenarios like cut-in, as a node can be in both sequences (imagine 2 targets performing cut-in one after the other
		
		if print_debug :
			print("\t07. scenario instance extraction..", end='')
			
		self.dataframe['scenario_label'] = self.dataframe['dictionary_index'].apply(lambda x : self.assign_scenario_label(x))
		
		tmpDf = self.dataframe.loc[self.dataframe['scenario_label'].diff() != 0]
		no_dup_dataframe = pd.DataFrame()
		no_dup_dataframe['df_idx'] = list(tmpDf.index)
		no_dup_dataframe['scenario_idx'] = list(tmpDf['scenario_label'])
		no_dup_dataframe['scenario_str'] = [self.scenario_indices[x] for x in  list(no_dup_dataframe['scenario_idx'])]
		no_dup_dataframe['dic_idx'] = list(tmpDf['dictionary_index'])

		#len(no_dup_dataframe)
		scenario_sequence = ''.join(no_dup_dataframe['scenario_str'])
		
		scenario_instances = [m.start() for m in re.finditer('psp', scenario_sequence)]
		
		if print_debug :
			print(len(scenario_instances), "instances..")
			
		self.dataframe['dictionary_index_int'] = self.dataframe['dictionary_index'].apply(lambda x : int(x))
		
		if print_debug :
			print("\t08. saving..", end='')
		
		scenario_retrieval_df = pd.DataFrame(columns=['init_pre_idx', 'init_scenario_idx', 'init_post_idx', 'end_post_idx',
												  'init_pre_ts', 'init_scenario_ts', 'init_post_ts', 'end_post_ts',
												  'pre_scenario_post_dict_sequence', 'scenario_dict_sequence'])
		for si in scenario_instances :
			mini_df = no_dup_dataframe.loc[si:si+len('sps')]
			init_pre_idx = mini_df.iloc[0]['df_idx']
			init_scenario_idx = mini_df.iloc[1]['df_idx']
			init_post_idx = mini_df.iloc[2]['df_idx']
			end_post_idx = mini_df.iloc[3]['df_idx']
			
			init_pre_ts = float(self.dataframe.loc[init_pre_idx, 'timestamp'])
			init_scenario_ts = float(self.dataframe.loc[init_scenario_idx, 'timestamp'])
			init_post_ts = float(self.dataframe.loc[init_post_idx, 'timestamp'])
			end_post_ts = float(self.dataframe.loc[end_post_idx, 'timestamp'])
			
			tmp_df = self.dataframe.loc[init_pre_idx:end_post_idx-1]
			dict_seq = list(tmp_df.loc[tmp_df['dictionary_index_int'].diff() != 0]['dictionary_index'])
			dict_seq_full = '-'.join(dict_seq)
			dict_seq_str = '-'.join(dict_seq[1:-1])
			
			scenario_retrieval_df = scenario_retrieval_df.append({'init_pre_idx' : init_pre_idx, 
																  'init_scenario_idx' : init_scenario_idx, 
																  'init_post_idx' : init_post_idx, 
																  'end_post_idx' : end_post_idx,
																  'init_pre_ts' : init_pre_ts, 
																  'init_scenario_ts' : init_scenario_ts, 
																  'init_post_ts' : init_post_ts, 
																  'end_post_ts' : end_post_ts,
																  'pre_scenario_post_dict_sequence' : dict_seq_full,
																  'scenario_dict_sequence' : dict_seq_str
																 }, ignore_index=True)
			#break


		scenario_retrieval_df.to_csv(self.output_path + "scenario_instances.csv", index=False)
		
		if print_debug :
			print("OK", end='')
			
		return scenario_retrieval_df
			
	def get_scenario_subnetwork_img_node(self, dataset_subnetwork_img, lbl) :
		node = None
		for n in dataset_subnetwork_img.nodes :
			if n['label'] == lbl :
				node = n
				break
		return node
		
	def generate_last_network(self, dataset_subnetwork_img, scenario_retrieval_df, print_debug) :
	
		if print_debug :
			print("\t09. generating scenario network..", end='')
			
		scenario_subnetwork_img = Network(height=500, width=1200, directed=True, notebook=False)
		
		added_nodes = []
		for i in range(len(scenario_retrieval_df)) :
			#print(i, '/', len(scenario_retrieval_df), end='\r')
			
			seq = scenario_retrieval_df.iloc[i]['pre_scenario_post_dict_sequence']
			seq = seq.split('-')
			
			# add nodes
			for j in range(len(seq)) :
				
				node_label = seq[j]
				if node_label not in added_nodes :
					# new node to add
					added_nodes.append(node_label)
					# get the style and augment
					node = self.get_scenario_subnetwork_img_node(dataset_subnetwork_img, node_label)
					colour = 'green'
					if j == 0 or j == len(seq) - 1 :
						colour = 'blue'
				
					node['color'] = {'border' : colour, 'highlight' : colour}
					node['borderWidth'] = 10
					node['borderWidth'] = 4
					node['borderWidthSelected'] = 4
					node['shapeProperties'] = {'useBorderWithImage' : True}
					
					scenario_subnetwork_img.add_node(int(node_label))
					
					for x in scenario_subnetwork_img.nodes :
						if x['id'] == int(node_label) :
							for k in node.keys() :
								x[k] = node[k]
					
				#break
			
			# add edges
			for j in range(len(seq) - 1) :
				from_node = int(seq[j])
				to_node = int(seq[j+1])
				found = False
				for e in scenario_subnetwork_img.edges :
					if e['from'] == from_node and e['to'] == to_node :
						found = True
						break
				   
				if not found :
					# we can add an edge
					scenario_subnetwork_img.add_edge(from_node, to_node)
					for i in range(len(scenario_subnetwork_img.edges)) :
						if scenario_subnetwork_img.edges[i]['from'] == from_node and scenario_subnetwork_img.edges[i]['to'] == to_node :
							# fix style
							scenario_subnetwork_img.edges[i]['color'] = edgeProperties['color']
					
				#break
			
				
			#break
		if print_debug :
			print("OK")


		scenario_subnetwork_img.save_graph(self.scenario_subnetwork_img_filename_stem + ".html")
		
		return scenario_subnetwork_img
		
	def patch_network_filenames(self, filename_stem) : 
		# output_path
		# inputPath_dictionaryImg
		# baseFolder
		fromPath = self.inputPath_dictionaryImg.replace('\\', '\\\\')
		toPath = "..\\\\" + "..\\\\" + self.inputFolder_dictionary + "\\\\" + self.inputFolder_dictionaryImg + "\\"
		
		with open(filename_stem + ".html", "r") as f:
			content = f.readlines()
		
		newContent = []
		for x in content :
			y = x
			while fromPath in y :
				#print(x)
				y = y[:y.find(fromPath)] + toPath + y[y.find(fromPath) + len(fromPath) : ]
			newContent.append(y)

		#for x in newContent :
		#	if fromPath in x :
		#		print(x)
		with open(self.output_path + filename_stem + ".html", "w") as f :
			f.writelines("".join(newContent))		
		
	def final_patch(self, print_debug) :
		if print_debug :
			print("\t10. final patch and network saving..", end='') 
			
		for f in [self.dataset_subnetwork_filename_stem, self.dataset_subnetwork_img_filename_stem, self.full_dataset_network_filename_stem, self.scenario_subnetwork_img_filename_stem] :
			self.patch_network_filenames(f)
			
	def braking_in_front_mining(self, file_to_parse, print_debug) :
		self.load_dictionary(print_debug)
		dict_valid_scenario_nodes = self.extract_breaking_in_front_nodes(print_debug)
		self.load_dataset(file_to_parse, print_debug)
		self.df_scenario_nodes, self.df_pre_post_nodes, self.df_remaining_nodes = self.dataset_node_identification(dict_valid_scenario_nodes, print_debug)
		
		full_dataset_network, dataset_subnetwork, dataset_subnetwork_img = self.generate_first_networks(print_debug)
		
		scenario_retrieval_df = self.scenario_instance_extraction(print_debug)
		scenario_subnetwork_img = self.generate_last_network(dataset_subnetwork_img, scenario_retrieval_df, print_debug)
		self.final_patch(print_debug)
		
			
	def run(self, file_to_parse, print_debug) :
	
		if print_debug :
			print("\n** BRAKING IN FRONT MINING **")
		
		if file_to_parse == 'all' :
			file_list = os.listdir(self.inputPath_data)

			for f in file_list :
				self.braking_in_front_mining(f, print_debug)
		else :
			self.braking_in_front_mining(file_to_parse, print_debug)
			
		if print_debug :
			print("\n** BYE! **")
			
			
			
if __name__ == "__main__" :

	arg_list =[{'arg_name' : 'file_to_parse', 'descr' : "either filename or 'all'"},
				{'arg_name' : 'print_debug', 'descr' : '0 for false, 1 for true'}]
	
	if len(sys.argv[1:]) != len(arg_list) :
		print("arg list:")
		for i in range(len(arg_list)) :
			print(i, 4*' ', arg_list[i]['arg_name'], ':', arg_list[i]['descr'])
	
	else :	
		arguments = ['script_name'] + [x['arg_name'] for x in arg_list]
		
		filename = sys.argv[arguments.index('file_to_parse')]
		print_debug = int(sys.argv[arguments.index('print_debug')])	

		BIF = braking_in_front_subgraph_scenario_extraction(filename, print_debug)
		BIF.run(filename, print_debug)

