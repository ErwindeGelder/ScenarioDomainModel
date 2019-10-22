import sys
import os

from dictionary_update_OO import dictionary_update
from dataset_mapping_OO import dataset_mapping
from braking_in_front_subgraph_scenario_extraction_OO import braking_in_front_subgraph_scenario_extraction

from imports.general_settings import *

def run(filename, print_debug) :
	DU = dictionary_update(print_debug) 
	DM = dataset_mapping(print_debug) 
	BIF = braking_in_front_subgraph_scenario_extraction(filename, print_debug) 

	DU.run(filename, print_debug)
	DM.run(filename, print_debug)
	BIF.run(filename, print_debug)	

def full_pipeline(raw_filename, print_debug) :

	data_folder = '..\\..\\data\\'
	base_folder = data_folder + projectName + '\\'
	input_folder = "00_raw_data"
	input_path = base_folder + input_folder

	if raw_filename == 'all' :
		file_list = [x for x in os.listdir(input_path) if 'hdf5' in x]

		for f in file_list :
			run(f, print_debug)
			#print(f)
			
	else :
		run(raw_filename, print_debug)
		

if __name__ == "__main__" :
	arg_list = [
				{'arg_name' : 'hdf5 raw file to parse', 'descr' : "either filename or 'all'"},
				#{'arg_name' : 'scenario name', 'descr' : "see general_settings.py for name", 'default' : 'breaking_in_front_close'},
				#{'arg_name' : 'scenario index', 'descr' : 'integer, usually 0, see general_settings.py', 'default' : 0},
				{'arg_name' : 'print debug', 'descr' : '0 for false, 1 for true', 'default' : 1}
				]
				
	n_compulsory = sum(['default' not in x for x in arg_list])

	if len(sys.argv[1:]) < n_compulsory or len(sys.argv[1:]) > len(arg_list) :
		print("\narg list:")
		for i in range(len(arg_list)) :
			print(i, 4*' ', arg_list[i]['arg_name'], ':', arg_list[i]['descr'], end=' ')
			if 'default' in arg_list[i] :
				print("( default:", arg_list[i]['default'], ")")
			else :
				print()
	
	else :	
		arguments = ['script_name'] + [x['arg_name'] for x in arg_list]
		def_values = [None] + [arg_list[i]['default'] if 'default' in arg_list[i] else None for i in range(len(arg_list))]
		values = [ sys.argv[i] if i < len(sys.argv) else None for i in range(len(arguments))]
		
		v = [ values[i] if values[i] != None else def_values[i] for i in range(len(values))]
		#print(v)
		
		raw_filename = v[arguments.index('hdf5 raw file to parse')]
		#scenario_name = v[arguments.index('scenario name')]
		#scenario_index = v[arguments.index('scenario index')]
		print_debug = int(v[arguments.index('print debug')])

		print("filename:", raw_filename)
		#print("scenario name:", scenario_name)
		#print("scenario index:", scenario_index)
		print("print debug:", print_debug)
		
		full_pipeline(raw_filename, print_debug)

	