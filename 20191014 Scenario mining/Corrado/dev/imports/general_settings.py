# this file contains settings/parameters/etc which are shared across notebooks

###########################################################################################################################
# PROJECT FOLDER/FILE SETTINGS
###########################################################################################################################

projectName = "ngg_jeroen_full_data"
nGram_n = 3

###########################################################################################################################
# VALID RELATIVE POSITIONS
###########################################################################################################################

valid_relative_positions = {
    'longitudinal' : ['x<-50',
                      '-50<=x<-30', 
                      '-30<=x<-10', 
                      '-10<=x<0',
                      '0<=x<=10', 
                      '10<x<=30', 
                      '30<x<=50',
                      'x>50',
                      ''],
    'lateral' : [#'left_of_left_lane', 
                 'left_lane',
                 'same_lane', 
                 'right_lane', 
                 #'right_of_right_lane',  
                 '']
}

n_targets = 8

###########################################################################################################################
# SCENARIO TEMPLATES
###########################################################################################################################

scenario_set = {
	'breaking_in_front_close' : [
							[{'target_relative_position_lateral' : 'same_lane',
							  'target_relative_position_longitudinal' : '0<=x<=10', 'target_longitudinal' : 'd'}]
						],
	
	'breaking_in_front_far' : [
							[{'target_relative_position_lateral' : 'same_lane',
							  'target_relative_position_longitudinal' : '10<x<=30', 'target_longitudinal' : 'd'}]
						],
}


###########################################################################################################################
# DICTIONARY COLUMNS
###########################################################################################################################

core_dictionary_columns = ['host_lateral', 'host_longitudinal'] + \
                          ['target_' + str(x) + "_" + y for x in range(8) for y in ['lateral', 
                                                                              'longitudinal', 
                                                                              'relative_position_longitudinal',
                                                                              'relative_position_lateral',
                                                                              'velocity'] ]

#img_dictionary_columns = ['n_objects', 'ego_img'] + \
#                         ['target_' + str(x) + "_img" for x in range(8)] + \
#                         ['full_img', 'ego_issue'] + \
#                         ['target_' + str(x) + "_issue" for x in range(8)]
img_dictionary_columns = ['n_objects', 'ego_issue'] + ['target_' + str(x) + "_issue" for x in range(8)]


dictionary_columns = ['index'] + core_dictionary_columns + img_dictionary_columns                    

###########################################################################################################################
# RENDERING COLOURS
###########################################################################################################################

BG_COLOUR = (122, 122, 122)
H_LANE_COLOUR = (255, 255, 255)
V_LANE_COLOUR = (0, 255, 255)

EGO_ACCELERATING_COLOUR = (0, 125, 0)
EGO_DECELERATING_COLOUR = (255, 0, 0)
EGO_CRUISING_COLOUR = (0, 0, 255)
EGO_BLINKER_COLOUR = (255, 255, 0)  # i.e. lateral movement
EGO_ISSUE = (255, 0, 255)

TARGET_ACCELERATING_COLOUR = (0, 125, 0)
TARGET_DECELERATING_COLOUR = (255, 0, 0)
TARGET_CRUISING_COLOUR = (0, 0, 255)
TARGET_BLINKER_COLOUR = (255, 255, 0)
TARGET_REL_SPEED_COLOUR = (0, 0, 0)
TARGET_ISSUE = (255, 0, 255)

###########################################################################################################################
# RENDERING SHAPE SIZES
###########################################################################################################################

EGO_IMG_W = 16
EGO_IMG_H = 8

TARGET_IMG_W = 12
TARGET_IMG_H = 8

# this is the amount of road around EGO/target, not the "cell" as the set of targets
CELL_IMG_W = 20
CELL_IMG_H = 12

BLINKER_W = 4
BLINKER_H = 4

H_LANE_H = 4
H_LANE_W = 8

V_LANE_H = 2
V_LANE_W = 2

MACRO_CELL_W = 2 * CELL_IMG_W # 2 columns
MACRO_CELL_H = 4 * CELL_IMG_H # 4 columns => 8 targets

###########################################################################################################################
# SCENARIO SUBNETWORK NODE RENDERING STYLE
###########################################################################################################################

general_nodeProperties = {
    'borderWidth' : 4,
    'borderWidthSelected' : 4,
    'shapeProperties' : {'useBorderWithImage' : True},
    'size' : 10
}

outScenario_nodeProperties = {
    'color' : {'border' : '#00aa00', 'highlight' : '#00aa00'}
}

scenario_nodeProperties = {
    'color' : {'border' : '#000000', 'highlight' : '#000000'}
}

edgeProperties = {
    'arrows' : 'to',
    'color' : {'color' :'#000000', 'highlight' : '#000000'}   
}

yMultiplier = 100

