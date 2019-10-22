#!/usr/bin/env python
# coding: utf-8

# # INTRO
# we map a dataset with the dictionary and also extract all relevant information for graph construction

# # HEADER
# adding a link to a shared folder where to find the ngram class

# In[154]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio
import sys

# In[2]:


from imports.general_settings import *
from imports.general_functions import *
from imports.NGram_KNrealRecursion import recursive_NGramKneserNey

# # IMPORTS

# In[155]:


import pickle
import networkx as nx


class dataset_mapping(object) :

	def __init__(self, print_debug) :
		self.file_and_folder_preparation(print_debug)
		
	def file_and_folder_preparation(self, print_debug=False) :
		data_folder = '..\\..\\data\\'
		base_folder = data_folder + projectName + '\\'
	
		inputFolder_data = "00_raw_data"
		inputFolder_dictionary = "01_activity_dictionary"
		outputFolder = "02_mapped_dataset"

		outputSubfolder = str(len(valid_relative_positions['lateral'])-1) + "x" + str(len(valid_relative_positions['longitudinal'])-1)

		self.outputFilename_dataset_stem = "mappedDataset"
		self.outputFilename_nGramGraph_stem = "nGramGraph"
		self.outputFilename_networkX_stem = "networkx"

		self.dictionaryFilename = "activityDictionary_" + str(len(valid_relative_positions['lateral'])-1) + "x" + str(len(valid_relative_positions['longitudinal'])-1) + ".csv"
		self.dictionary_indexColumnName = "index"

		self.inputPath_data = base_folder + inputFolder_data + "\\"
		self.inputPath_dictionary = base_folder + inputFolder_dictionary + "\\"
		self.outputPath = base_folder + outputFolder + "\\" + outputSubfolder + "\\"

		if print_debug :
			print("** FILE AND FOLDER SETTINGS **")
			print("\tbase_folder:", base_folder)
			print("\tinputFolder_data:", inputFolder_data)
			print("\tinputFolder_dictionary:", inputFolder_dictionary)
			print("\toutputFolder:", outputFolder)
			print("\toutputSubfolder:", outputSubfolder)
			print("\toutputFilename_dataset_stem:", self.outputFilename_dataset_stem)
			print("\toutputFilename_nGramGraph_stem:", self.outputFilename_nGramGraph_stem)
			print("\toutputFilename_networkX_stem:", self.outputFilename_networkX_stem)
			print("\tdictionaryFilename:", self.dictionaryFilename)
			print("\tdictionary_indexColumnName:", self.dictionary_indexColumnName)
			print("\tinputPath_data:", self.inputPath_data)
			print("\tinputPath_dictionary:", self.inputPath_dictionary)
			print("\toutputPath:", self.outputPath)
			#print("\t:",)
			#print("\t:",)
			#print("\t:",)
			
		if print_debug :
			print("\n** FOLDER INITIALISATIONS **")

		if outputFolder not in os.listdir(base_folder) :
			print("creating", base_folder + outputFolder)
			os.mkdir(base_folder + outputFolder)
		else :
			if print_debug :
				print("\tpath", base_folder + outputFolder, "already exists")
	
		if outputSubfolder not in os.listdir(base_folder + outputFolder) :
			print("creating", self.outputPath)
			os.mkdir(self.outputPath)
		else :
			if print_debug :
				print("\tpath", self.outputPath, "already exists")
				
	def load_dataset(self, inputFile, print_debug) :
	
		if print_debug :
			print("\t02. loading", inputFile, "..", end='')
			
		s = pd.HDFStore(self.inputPath_data + inputFile)
		tagged_dataset = s.get('df')
		s.close()
		tagged_dataset.reset_index(inplace=True, drop=True)
	
		if print_debug :
			print("OK, shape =", tagged_dataset.shape)
			
		# # TAGGED DATAFRAME CORRECTION BASED ON VALID RELATIVE POSITIONS
		if print_debug :
			print("\t03. dataset correction based on dictionary grid-size..", end='')
		
		for t in range(n_targets) :
			tagged_dataset.loc[(~tagged_dataset['target_' + str(t) + '_relative_position_longitudinal'].isin(valid_relative_positions['longitudinal'])) |
				  (~tagged_dataset['target_' + str(t) + '_relative_position_lateral'].isin(valid_relative_positions['lateral'])),
							['target_' + str(t) + '_lateral', 
							 'target_' + str(t) + '_longitudinal',
							 'target_' + str(t) + '_relative_position_longitudinal', 
							 'target_' + str(t) + '_relative_position_lateral',
							 'target_' + str(t) + '_velocity']] = ''
		if print_debug :
			print("OK")
			
		return tagged_dataset
		
	def load_dictionary(self, print_debug) :
	
		if print_debug :
			print("\t01. loading dictionary", self.dictionaryFilename, "..", end='')
	
		self.dictionary = pd.read_pickle(self.inputPath_dictionary + self.dictionaryFilename)
		
		if print_debug :
			print("OK, shape =", self.dictionary.shape)
				
	def map_dataset(self, file_to_parse, print_debug) :

		self.load_dictionary(print_debug)
		
		tagged_dataset = self.load_dataset(file_to_parse, print_debug)
		
		if print_debug :
			print("\t04. mapping..", end='')

		shared_columns = []
		for c in tagged_dataset.columns :
			if c in self.dictionary.columns :
				shared_columns.append(c)
				
		#if print_debug :
		#	print("\t\tshared columns:", shared_columns)
			
		merged_dataset = tagged_dataset.merge(self.dictionary, on=shared_columns)
		merged_dataset.sort_values(by=['timestamp'], ascending=[True], inplace=True)
		
		if sum(np.asarray(list(merged_dataset['timestamp'])) == np.asarray(list(tagged_dataset['timestamp']))) == len(tagged_dataset) :
			if print_debug :
				print("OK")
		else :
			print("ERROR")
			return
			
		if print_debug :
			print("\t05. cleaning up..", end='')
		cols_to_remove = []
		for c in merged_dataset.columns :
			if c not in [self.dictionary_indexColumnName, 'n_objects'] + list(tagged_dataset.columns) :
				cols_to_remove.append(c)

		merged_dataset.drop(cols_to_remove, axis=1, inplace=True)
		merged_dataset.rename(columns={self.dictionary_indexColumnName : 'dictionary_index'}, inplace=True)

		if print_debug :
			print("OK")
			
		if print_debug :
			print("\t06. saving merged dataset..", end='')
			
		outputFilename_dataset = self.outputFilename_dataset_stem + "_" + file_to_parse[:file_to_parse.find(".hdf5")] + ".csv"
		merged_dataset.to_csv(self.outputPath + outputFilename_dataset, index=False)
		
		if print_debug :
			print("OK")	
			
		if print_debug :
			print("\t07. n-gram creation..", end='')		

		# # N-GRAM GRAPH CREATION
		# we build a unique ngram, the rendering/filtering should occur later

		model = recursive_NGramKneserNey("world_model_" + str(len(valid_relative_positions['lateral'])-1) + "x" + str(len(valid_relative_positions['longitudinal'])-1), n=nGram_n)
		string = "-".join([str(x) for x in list(merged_dataset['dictionary_index'])])
		model.feed(string)

		outputFilename_nGramGraph = self.outputFilename_nGramGraph_stem + "_" + file_to_parse[:file_to_parse.find(".hdf5")] + ".p"
		pickle.dump(model, open(self.outputPath + outputFilename_nGramGraph, 'wb'))

		if print_debug :
			print("OK")	
			
		if print_debug :
			print("\t08. graph creation..", end='')

		# # NETWORKX CREATION
		# i.e. a network from nGramGraph without self loops and <start><stop><unk>

		G = nx.DiGraph()
		for edge in model.grams[2] :
			n0 = edge[0]
			n1 = edge[1]
			if n0 in ["<start>", "<stop>", "<unk>"] or n1 in ["<start>", "<stop>", "<unk>"] or n0 == n1:
				#if n0 != n1 :
				#	print(edge)
				continue
			G.add_edge(n0, n1)
		#print(".")

		#G.number_of_edges()
		#G.number_of_nodes()

		outputFilename_networkX = self.outputFilename_networkX_stem + "_" + file_to_parse[:file_to_parse.find(".hdf5")] + ".p"
		pickle.dump(G, open(self.outputPath + outputFilename_networkX, "wb"))

		if print_debug :
			print("OK")	
				
				
	def run(self, file_to_parse, print_debug) :
		
		if print_debug :
			print("\n** DATASET MAPPING **")
		
		if file_to_parse == 'all' :
			file_list = os.listdir(self.inputPath_data)

			for f in file_list :
				self.map_dataset(f, print_debug)
		else :
			self.map_dataset(file_to_parse, print_debug)
			
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

		DM = dataset_mapping(print_debug) 
		DM.run(filename, print_debug)
	


