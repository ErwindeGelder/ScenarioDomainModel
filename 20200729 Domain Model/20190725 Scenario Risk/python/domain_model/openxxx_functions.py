"""
OpenXXX_functions:

Functions that support the functions "to_openscenario" of scenario class and "to_opendrive" from
static_environment class

Author
------
Jeroen Broos

Creation
--------
27 Nov 2018

To do
-----

Modifications
-------------

"""

import xml.etree.ElementTree as ET
import xml.dom.minidom
import datetime

def set_general_elements(scenario, filename, open_scenario):
    """ Writes general elements of OpenSCENARIO code:
            - FileHeader
            - RoadNetwork inclusive Logics

        :param scenario: description of scenario in domain model
        :param filename: name of OpenSCENARIO file
        :param open_scenario: parent of xml element
        """

    ET.SubElement(open_scenario, 'FileHeader',
                  revMajor="1",
                  revMinor="0",
                  date=datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                  description=scenario.name,
                  author="TNO StreetWise Database")
    ET.SubElement(open_scenario, 'ParameterDeclaration')

    catalogs = ET.SubElement(open_scenario, 'Catalogs')
    vehicle_catalog = ET.SubElement(catalogs, 'VehicleCatalog')
    ET.SubElement(vehicle_catalog, 'Directory', path="")
    driver_catalog = ET.SubElement(catalogs, 'DriverCatalog')
    ET.SubElement(driver_catalog, 'Directory', path="")
    pedestrian_catalog = ET.SubElement(catalogs, 'PedestrianCatalog')
    ET.SubElement(pedestrian_catalog, 'Directory', path="")
    pedestrian_controller_catalog = ET.SubElement(catalogs, 'PedestrianControllerCatalog')
    ET.SubElement(pedestrian_controller_catalog, 'Directory', path="")
    misc_object_catalog = ET.SubElement(catalogs, 'MiscObjectCatalog')
    ET.SubElement(misc_object_catalog, 'Directory', path="")
    environment_catalog = ET.SubElement(catalogs, 'EnvironmentCatalog')
    ET.SubElement(environment_catalog, 'Directory', path="")
    maneuver_catalog = ET.SubElement(catalogs, 'ManeuverCatalog')
    ET.SubElement(maneuver_catalog, 'Directory', path="")
    trajectory_catalog = ET.SubElement(catalogs, 'TrajectoryCatalog')
    ET.SubElement(trajectory_catalog, 'Directory', path="")
    route_catalog = ET.SubElement(catalogs, 'RouteCatalog')
    ET.SubElement(route_catalog, 'Directory', path="")

    road_network = ET.SubElement(open_scenario, 'RoadNetwork')
    ET.SubElement(road_network, 'Logics',
                  filepath=filename + '_' + scenario.static_environment.name.replace(" ", "_") + '.xodr')
    ET.SubElement(road_network, 'SceneGraph', filepath="")

def set_entity(actor, entity):
    """ Writes OpenSCENARIO code for entity

        Limitations:
            - Only category "CategoryM_PassengerCar" and "CategoryN_LGV"
            - Standard dimensions and dynamics parameters

        :param actor: entities that have to be described
        :param entities: xml parent field
        :return: object_element: XML tree of the objects that are part of the entity
        """

    object_element = ET.SubElement(entity, 'Object', name=actor.name)
    if str(actor.actor_category.vehicle_type) == "VehicleType.Vehicle" or \
                    str(actor.actor_category.vehicle_type) == "VehicleType.CategoryM_PassengerCar":
        vehicle = ET.SubElement(object_element, 'Vehicle', name=actor.name + '_car',
                                category="car")
        ET.SubElement(vehicle, 'Performance', maxSpeed="50", maxDeceleration="9.5", mass="1438.5")
        bounding_box = ET.SubElement(vehicle, 'BoundingBox')
        ET.SubElement(bounding_box, 'Center', x="1.366", y="0", z="0.721")
        ET.SubElement(bounding_box, 'Dimension', width="1.799", length="4.258",
                      height="1.452")
        axles = ET.SubElement(vehicle, 'Axles')
        ET.SubElement(axles, 'Front', maxSteering="0.48", wheelDiameter="0.671", trackWidth="1.549",
                      positionX="2.630", positionZ="0.300")
        ET.SubElement(axles, 'Rear', maxSteering="0", wheelDiameter="0.600", trackWidth="1.533",
                      positionX="0", positionZ="0.3355")
        Properties = ET.SubElement(vehicle, 'Properties')
        if "Tag.EgoVehicle" not in str(actor.tags):
            ET.SubElement(Properties, 'Property', name="control", value="internal")

    elif str(actor.actor_category.vehicle_type) == "VehicleType.CategoryN_LGV":
        vehicle = ET.SubElement(object_element, 'Vehicle', name=actor.name,
                                category="truck")
        ET.SubElement(vehicle, 'Performance', maxSpeed="27.8", maxDeceleration="8.0", mass="10840")
        bounding_box = ET.SubElement(vehicle, 'BoundingBox')
        ET.SubElement(bounding_box, 'Center', x="2.6075", y="0", z="1.8975")
        ET.SubElement(bounding_box, 'Dimension', width="2.495", length="11.985",
                      height="3.795")
        axles = ET.SubElement(vehicle, 'Axles')
        ET.SubElement(axles, 'Front', maxSteering="0.48", wheelDiameter="0.980", trackWidth="2.060",
                      positionX="7.125", positionZ="0.490")
        ET.SubElement(axles, 'Rear', maxSteering="0", wheelDiameter="0.980", trackWidth="1.835",
                      positionX="0", positionZ="0.490")
        Properties = ET.SubElement(vehicle, 'Properties')
        if "Tag.EgoVehicle" not in str(actor.tags):
            ET.SubElement(Properties, 'Property', name="control", value="internal")

    return object_element

def set_weather_conditions(actions):
    """ Writes OpenSCENARIO code for Global environment

            Limitations:
                - For every scenario the same (not yet defined in domain model)

            :param actions: xml parent field
            :return: global: XML tree of the objects that are part of the global environment
            """

    global_env = ET.SubElement(actions, 'Global')
    set_environment = ET.SubElement(global_env, 'SetEnvironment')
    environment = ET.SubElement(set_environment, 'Environment', name="GlobalEnvironment")
    time_of_day = ET.SubElement(environment, 'TimeOfDay', animation="false")
    ET.SubElement(time_of_day, 'Time', hour="12", min="0", sec="0.0")
    ET.SubElement(time_of_day, 'Date', day=datetime.datetime.now().strftime('%d'),
                  month=datetime.datetime.now().strftime('%m'),
                  year=datetime.datetime.now().strftime('%Y'))
    weather = ET.SubElement(environment, 'Weather', cloudState="free")
    ET.SubElement(weather, 'Sun', intensity="100000", azimuth="0", elevation="1.571")
    ET.SubElement(weather, 'Fog', visualRange="100000")
    ET.SubElement(weather, 'Precipitation', type="dry", intensity="0")
    ET.SubElement(environment, 'RoadCondition', frictionScale="1")

    return global_env

def set_initial_status_actors(scenario, actor, actions):
    """ Writes OpenSCENARIO code for entity

        Limitations:
            -

        :param scenario: scenario described with domain model
        :param actor: actor for which the initial status has to be described
        :param actions: xml parent field
        :return: private: XML tree with intial status information of the actor
        """

    private = ET.SubElement(actions, 'Private', object=actor.name)
    action = ET.SubElement(private, 'Action')
    position = ET.SubElement(action, 'Position')
    pos, vel, ori = initial_state(scenario, actor.name)
    lane = ET.SubElement(position, 'Lane',
                         roadId=str(pos[0]),
                         laneId=str(pos[1]),
                         s=str(pos[2]),
                         offset=str(pos[3]))
    ET.SubElement(lane, 'Orientation', type="absolute",
                  h=str(ori),
                  p="0",
                  r="0")
    action = ET.SubElement(private, 'Action')
    longitudinal = ET.SubElement(action, 'Longitudinal')
    speed = ET.SubElement(longitudinal, 'Speed')
    ET.SubElement(speed, 'Dynamics', shape="step", rate="0")
    target = ET.SubElement(speed, 'Target')
    ET.SubElement(target, 'Absolute', value=str(vel))

    return private

def set_act(scenario, actor, story):
    """ Writes OpenSCENARIO code for acts

        Limitations:
            -

        :param scenario: scenario described with domain model
        :param actor: actor for which the act has to be described
        :param story: xml parent field
        :return: act: XML tree that describes the act
        """

    act = ET.SubElement(story, 'Act', name=actor.name + "_acts")
    sequence = ET.SubElement(act, 'Sequence', name=actor.name + "_sequence",
                             numberOfExecutions="1")
    actors = ET.SubElement(sequence, 'Actors')
    ET.SubElement(actors, 'Entity', name=actor.name)

    # Define one maneuver for each actor
    maneuver = ET.SubElement(sequence, 'Maneuver', name=actor.name + " maneuver")
    for acts in scenario.acts:
        if actor.name in acts[0].name:
            set_event(actor, acts, maneuver)

    # Define start time of each sequence
    conditions = ET.SubElement(act, 'Conditions')
    start = ET.SubElement(conditions, 'Start')
    condition_group = ET.SubElement(start, 'ConditionGroup')
    condition = ET.SubElement(condition_group, 'Condition',
                              name=actor.name + " sequence start time", delay="0",
                              edge="rising")
    by_value = ET.SubElement(condition, 'ByValue')
    ET.SubElement(by_value, 'SimulationTime',
                  value=str(scenario.time["start"]),
                  rule="greater_than")

    return act

def set_event(actor, acts, maneuver):
    """ Writes OpenSCENARIO code for events

        Limitations:
            -

        :param actor: actor for which the maneuver has to be described
        :param acts: act that is described with the maneuver
        :param maneuver: xml parent field
        :return event: XML tree that describes the maneuver
        """

    event = ET.SubElement(maneuver, 'Event', name=acts[1].name + " event",
                          priority="overwrite")
    action = ET.SubElement(event, 'Action', name=acts[1].name + " action")
    private = ET.SubElement(action, 'Private')
    if isinstance(acts[1].parameters['xend'][0], str):
        if acts[1].parameters['xstart'][:7] == "CONTROL":
            ET.SubElement(private, 'Autonomous',
                          domain=acts[1].parameters['xstart'][8:],
                          activate="false")
        elif acts[1].parameters['xend'][:7] == "CONTROL":
            ET.SubElement(private, 'Autonomous',
                          domain=acts[1].parameters['xend'][8:],
                          activate="true")
    elif acts[1].activity_category.state.name == "LONGITUDINAL_POSITION" or \
                    acts[1].activity_category.state.name == "SPEED":
        longitudinal = ET.SubElement(private, 'Longitudinal')
        speed = ET.SubElement(longitudinal, 'Speed')
        target = ET.SubElement(speed, 'Target')
        if "Linear" in str(acts[1].activity_category.model):
            if acts[1].activity_category.state.name == "LONGITUDINAL_POSITION":
                actor_final_speed = (acts[1].parameters['xend'][1] -
                                     acts[1].parameters['xstart'][1]) / \
                                    acts[1].tduration
                act_rate = 0
            elif str(acts[1].activity_category.state) == "StateVariable.SPEED":
                actor_final_speed = acts[1].parameters['xend'][0]
                act_rate = abs((acts[1].parameters['xend'][0] - acts[1].parameters['xstart'][0]) / acts[1].tduration)
            ET.SubElement(target, 'Absolute',
                          value=str(actor_final_speed))
            ET.SubElement(speed, 'Dynamics', shape="linear",
                          rate=str(act_rate))
    elif acts[1].activity_category.state.name == "LATERAL_POSITION":
        lateral = ET.SubElement(private, 'Lateral')
        lanechange_direction = (acts[1].parameters['xend'][0] -
                                acts[1].parameters['xstart'][0])
        if lanechange_direction == 0:
            lane_offset = ET.SubElement(lateral, 'LaneOffset')
            target = ET.SubElement(lane_offset, 'Target')
            if "Linear" in str(acts[1].activity_category.model):
                act_rate = abs((acts[1].parameters['xend'][1] - acts[1].parameters['xstart'][1]) / acts[1].tduration)
                ET.SubElement(target, 'Absolute',
                              value=str(acts[1].parameters['xend'][1]))
                ET.SubElement(lane_offset, 'Dynamics', shape="linear",
                              maxLateralAcc=str(act_rate))
        elif lanechange_direction != 0:
            lane_change = ET.SubElement(lateral, 'LaneChange', targetLaneOffset="0")
            target = ET.SubElement(lane_change, 'Target')
            ET.SubElement(target, 'Relative', object=actor.name,
                          value=str(lanechange_direction))
            if "Sinusoidal" in str(acts[1].activity_category.model):
                ET.SubElement(lane_change, 'Dynamics',
                              shape="sinusoidal",
                              time=str(acts[1].tduration))

    # Define start time of each activity
    start_conditions = ET.SubElement(event, 'StartConditions')
    condition_group = ET.SubElement(start_conditions, 'ConditionGroup')
    condition = ET.SubElement(condition_group, 'Condition',
                              name=acts[1].name + " event start time", delay="0",
                              edge="rising")
    by_value = ET.SubElement(condition, 'ByValue')
    ET.SubElement(by_value, 'SimulationTime', value=str(acts[2]),
                  rule="greater_than")

    # Add additional trigger condition for TriggeredActivities
    if hasattr(acts[1], 'trigger'):
        condition = acts[1].trigger.conditions["Condition"]
        xml_condition = ET.fromstring(condition)
        condition_group.append(xml_condition)

    return event

def initial_state(scenario, actor):
    """ Determines the initial position of an actor in a scenario

    Limitations:
        - Only linear and sinusoid activity models supported
        - Only road elements with only one geometry element supported
        - Initial velocity only valid for:
            - straight roads
            - no lateral movement during first longitudinal activity
        - Initial orientation of actor assumed to be equal to road orientation
            - only straight (shape=line) roads supported

    :param scenario: scenario described with domain model
    :param actor: actor of interest
    :return: init_pos: Initial position of ACTOR [road_id, lane_id, start, offset]
    :return: init_vel: Initial velocity of ACTOR
    :return: init_ori: Initial orientation of ACTOR in world coordinate system
    """

    # Determine activities of actor at start of scenario
    for act in scenario.acts:
        if actor in act[0].name and act[2] == scenario.time['start']:
            # Initial road id and start are stored in "LONGITUDINAL_POSITION" state activities
            if str(act[1].activity_category.state) == "StateVariable.LONGITUDINAL_POSITION":
                road_id = act[1].parameters['xstart'][0]
                start = act[1].parameters['xstart'][1]

                # Initial velocity is determined from start and end parameters (no lateral movement
                # assumed)
                init_vel = (act[1].parameters['xend'][1] - act[1].parameters['xstart'][1]) / \
                           act[1].tduration
            # Initial lane id and offset are stored in "LATERAL_POSITION" state activities
            if str(act[1].activity_category.state) == "StateVariable.LATERAL_POSITION":
                lane_id = act[1].parameters['xstart'][0]
                offset = act[1].parameters['xstart'][1]
    init_pos = [road_id, lane_id, start, offset]

    # Based on initial position of actor the initial orientation is determined
    for road_ids in range(0, len(scenario.static_environment.properties['Roads'])):
        if scenario.static_environment.properties['Roads'][road_ids]['ID'] == road_id:
            init_ori = scenario.static_environment.properties['Roads'][road_ids]['Geometry'][0]['heading']
            break

    return init_pos, init_vel, init_ori
