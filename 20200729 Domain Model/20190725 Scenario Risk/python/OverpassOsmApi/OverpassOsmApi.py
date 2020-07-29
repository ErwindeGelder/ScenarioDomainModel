# todo: 1 plug back tictoc
import csv
from geopy.geocoders import Nominatim
import utm
import numpy as np
import overpy
import time
from scipy.interpolate import CubicSpline
from scipy.interpolate import interpolate
from math import log, exp, tan, atan, pi, ceil
from shapely import geometry as geom
import operator
#from blocks.srnn_pytorch.srnn.misc_utils import TicToc

class OverpassOsmApi(object):
    def __init__(self, url_list=['http://overpass-api.de/api/interpreter'],
                 url_list_file='overpass_servers.txt',
                 region_database='countries.csv',  # the path to the CSV with database ISO-3166-Countries-with-Regional-Codes, taken from: https://github.com/lukes/ISO-3166-Countries-with-Regional-Codes/blob/master/all/all.csv
                 buffered_arrea_bbox_margin=800,  # Margin used for the bounding box [m], 800 was optimal
                 max_dist_from_motorway=20):  # Maximum distance from motorway to be considered as on the motorway [m]
        self.region_database = region_database
        self.api = overpy.Overpass()
        # if the list is not provided explicitly -> read it from the file
        if not url_list:
            self.url_list = open(url_list_file, 'r').read().split('\n')
        self.url_list_len = len(self.url_list)
        self.url_pointer = -1
        self.tick_url_list()
        self.buffered_area_bbox_margin = buffered_arrea_bbox_margin  # the margins (1/2 of the box dim) for the buffered area (square)
        self.max_dist_from_motorway = max_dist_from_motorway
        self.result = None  # storage for the result retrieved from API
        self.utm = None  # storage for the coords in UTM
        self.margins = [0, 0, 0, 0]   # storage for the buffered area box (UTM)
        self.updated_result = None  # storage for the result (retrieved from API) and corresponding geometry
        self.distances = []  # the list of distances from a specific point to the geometry stored in self.updated_result
        self.sorted_distances = []  # indexes of the ways sorted by distance (self.distances)
        self.closest_ways = None  # the list of closest ways within max_dist_from_motorway
        self._time = None  # storage for tic/toc function
        self.cs_over_x = None
        self.cs_over_y = None
        # the lists of useful road and links names
        self.motorway_list = ['motorway', 'trunk', 'primary', 'secondary']
        self.motorway_link_list = ['motorway_link', 'trunk_link', 'primary_link', 'secondary_link']

    # stores current time
    def tic(self, num=0):
        self._time = time.time()
        return self._time[num]

    # tells how much time passed from the last self.tic()
    def toc(self):
        dtime = time.time() - self._time
        return dtime

    # tells how much time passed from the last self.tic() and resets the counter
    def retoc(self):
        ret = self.toc()
        self.tic()
        return ret

    # one tick of the revolver buffer for the list of overpass servers
    def tick_url_list(self):
        self.url_pointer +=1
        if self.url_pointer > self.url_list_len:
            self.url_pointer = 0
        self.api.url = self.url_list[self.url_pointer]

    # queries from the overpass all the data within the buffered area, sets up the area; uses a tag to limit the request
    def query(self, coords, tag='highway', is_utm=False):
        easting, northing, zone_number, zone_char = coords if is_utm else utm.from_latlon(coords[0], coords[1])
        # allocate and store the large box margins
        self.margins[0] = easting + self.buffered_area_bbox_margin
        self.margins[1] = easting - self.buffered_area_bbox_margin
        self.margins[2] = northing + self.buffered_area_bbox_margin
        self.margins[3] = northing - self.buffered_area_bbox_margin
        lb = utm.to_latlon(easting - self.buffered_area_bbox_margin, northing - self.buffered_area_bbox_margin, zone_number, zone_char)
        rt = utm.to_latlon(easting + self.buffered_area_bbox_margin, northing + self.buffered_area_bbox_margin, zone_number, zone_char)
        query = 'way('
        query += '{:.5f}, {:.5f}'.format(lb[0], lb[1])
        query += ', {:.5f}, {:.5f}'.format(rt[0], rt[1])
        query += ') [' + tag + '];(._;>;);out body;'
        # Retrieve result using API
        try:
            self.result = self.api.query(query)
        except:
            return None
        # create a shapely.geometry.line for every way and update the ways with it
        self.updated_result = []
        for way in self.result.ways:
            point_cloud = [utm.from_latlon(node.lat, node.lon)[0:2] for node in way.nodes]
            line = geom.LineString(point_cloud)
            self.updated_result.append([way, line])
        return self.updated_result


    # checks whether the given coordinates in the buffered area
    def whether_in_buffered_area_bbox(self, coords, is_utm=False):
        # checking that coords + min_box are inside the large box
        easting, northing, zone_number, zone_char = coords if is_utm else utm.from_latlon(coords[0], coords[1])
        # shrink the large box by max_dist_from_motorway
        point_easting_max = self.margins[0] - self.max_dist_from_motorway
        point_easting_min = self.margins[1] + self.max_dist_from_motorway
        point_nothing_max = self.margins[2] - self.max_dist_from_motorway
        point_nothing_min = self.margins[3] + self.max_dist_from_motorway
        # cheking itself
        if easting > point_easting_max:
            return False
        elif easting < point_easting_min:
            return False
        elif northing > point_nothing_max:
            return False
        elif northing < point_nothing_min:
            return False
        else:
            return True


    # makes the list of closest within max_dist_from_motorway ways to the given point
    def get_closest_ways(self, coords, is_utm=False, max_dist_from_motorway=None, with_distances=False):
        if not max_dist_from_motorway:
            max_dist_from_motorway = self.max_dist_from_motorway
        self.update_distances(coords, is_utm)
        return self.select_closest_ways(max_dist_from_motorway, with_distances=with_distances)


    # computes distances from the ways in buffered area to the provided coords, sorts the distances (closest first)
    def update_distances(self, coords, is_utm=False):
        easting, northing, zone_number, zone_char = coords if is_utm else utm.from_latlon(coords[0], coords[1])
        # update distances
        self.distances = []
        for way in self.updated_result:
            line = way[1]
            point = geom.Point(easting, northing)
            self.distances.append(line.distance(point))
        # sorting
        self.sorted_distances = np.argsort(self.distances)

        return self.sorted_distances


    # collects for all the ways that are closer than max_dist_from_motorway, if they are in our interest (motorway_list)
    def select_closest_ways(self, max_dist_from_motorway=None, with_distances=False, tag='highway'):
        # taking the first closest way than max_dist_from_motorway
        if not max_dist_from_motorway:
            max_dist_from_motorway = self.max_dist_from_motorway
        closest_ways = []
        for dst_idx in self.sorted_distances:
            if self.distances[dst_idx] < max_dist_from_motorway:
                way = self.updated_result[dst_idx][0]
                to_append = (way, self.distances[dst_idx]) if with_distances else way
                closest_ways.append(to_append)
            else:
                break
        return closest_ways


    # this function queries overpass if the coordinates are out of the buffered area; it output ways and distances list
    def find_ways_dst_from_wgs(self, coords):
        if not self.whether_in_buffered_area_bbox(coords):
            try:
                self.query(coords)
            except:
                return None
        self.closest_ways = self.get_closest_ways(coords, with_distances=True)
        return self.closest_ways

    # this is used to check how long time it takes to process different sizes of the buffered area, to find the best one
    # draft
    def _find_best_buffered_area_bbox_margin(self, coords, coords_2, margin_list=[60, 100, 500, 800, 1000, 1500, 2000, 3000]):
        #easting, northing, zone_number, zone_char
        for i in margin_list:  # , 5000, 10000, 15000]:
            self.buffered_area_bbox_margin = i
            searches = int(i / 30) * 4  # 30 m/s - ego velocity, 4Hz - GPS rate
            self.tic()
            self.query(coords, is_utm=False, motorway_list=self.motorway_list + self.motorway_link_list)
            A = self.retoc()
            for s in range(searches):
                way = self.find_right_road(None, coords, coords_2, is_utm=False)[0]
                self.get_road_config(way, coords, False)
            B = self.toc()
            print(i, ':', (A + B) * 60000 / i)  # normalize, time was in ms
        print('completed')


    # finds suitable routes based on the list of WGS coordinates
    def get_routes_from_coords_list(self, coords_list, verbose=False):
        # query OSM
        ways_dst_list = []
        for coords_idx, coords in enumerate(coords_list):
            success_query_flag = False
            for _ in range(self.url_list_len):
                try:
                    ways_dst_list.append(self.find_ways_dst_from_wgs(coords))
                    success_query_flag = True
                    break
                except:
                    self.tick_url_list()
                    if verbose:
                        print('server returned an error, new url:', self.url_list[self.url_pointer])
            if not success_query_flag:
                return coords_idx, None

        # list of coords -> list of [list of ways, acc distances, index counter]
        #   list of ways is a route candidate
        #   distances needed to estimate closest way
        #   index counter need to control losted ways (are not updated every time)
        routes = [[[way_dst[0]], way_dst[1], 0, 0] for way_dst in ways_dst_list[0]]

        for way_dst_idx, way_dst in enumerate(ways_dst_list[1:]):
            for (way_cand, dst_cand) in way_dst:
                for route_item_idx, route_item in enumerate(routes):
                    last_way_in_chain = route_item[0][-1]
                    # a new way, but connected to the last one - > split to a new branch + update distances, indexes
                    if self.check_ways_connection(last_way_in_chain, way_cand) == 1:
                        if routes[route_item_idx][3] == way_dst_idx + 1:
                            updated_way = 0
                        else:
                            updated_way = 1
                        way_branch = list(routes[route_item_idx][0])
                        way_branch.append(way_cand)
                        routes.append([way_branch, routes[route_item_idx][1] + dst_cand, routes[route_item_idx][2] + updated_way, way_dst_idx + 1])
                    # we are still on the same way -> update distances, indexes
                    elif last_way_in_chain.id == way_cand.id:
                        if routes[route_item_idx][3] == way_dst_idx + 1:  # if so we are on a just branched route
                            continue
                        routes[route_item_idx][1] += dst_cand
                        routes[route_item_idx][2] += 1
                        routes[route_item_idx][3] = way_dst_idx + 1
            # rid of losted ways
            routes = [chain for chain in routes if chain[2] == way_dst_idx + 1]
            # keep the closest 10 only, otherwise it grows too fast
            routes = sorted(routes, key=operator.itemgetter(1))[:10]

        # return ways only
        routes = [c[0] for c in routes]
        return routes


    def get_road_data_from_list_of_ways(self, chain):
        nodes = []  # the list of all nodes in the route
        road_data = []  # the list of road data segments
        road_data_candidate_old = None

        for way in chain:
            # append nodes coordinates
            for node in way.nodes:
                easting, northing, _, _ = utm.from_latlon(node.lat, node.lon)
                if nodes:
                    if nodes[-1] == [easting, northing]:
                        continue
                nodes.append([easting, northing])

            # read the way data
            lanes = int(way.tags.get('lanes'))
            oneway = way.tags.get('oneway')
            oneway = True if oneway == 'yes' else False
            road_type = way.tags.get('highway')
            maxspeed = self.try_int(way.tags.get('maxspeed'))
            maxspeed_lanes = self.tags_to_list(way.tags.get('maxspeed:lanes'))
            if maxspeed_lanes:
                maxspeed_lanes = [self.try_int(subitem) for item in maxspeed_lanes for subitem in item]
            turn_lanes = self.tags_to_list(way.tags.get('turn:lanes'))
            if not turn_lanes:
                turn_lanes = [['none']] * lanes
            turn_lanes_forward = way.tags.get('turn:lanes:forward')
            turn_lanes_backward = way.tags.get('turn:lanes:backward')
            s_coord = len(nodes)
            road_data_candidate = {'road_type': road_type, 'lanes': lanes, 'oneway': oneway, 'turn_lanes': turn_lanes,
                                   'turn_lanes_forward': turn_lanes_forward, 'turn_lanes_backward': turn_lanes_backward,
                                   'maxspeed': maxspeed, 'maxspeed_lanes': maxspeed_lanes, 's_coord': s_coord,
                                   's_length': 0}
            # if data is duplicated -> keep the same data
            # overuse s_coord
            if road_data_candidate_old:
                #if road_data_candidate['s_coord'] != road_data_candidate_old['s_coord']:
                if not self.equal_dicts(road_data_candidate, road_data_candidate_old, ['s_coord']):
                    road_data.append(road_data_candidate_old)
            road_data_candidate_old = road_data_candidate

        # the last one does not satisfy the condition above but must be included
        road_data.append(road_data_candidate_old)

        return np.array(nodes), road_data

    # approximately fits cubic spline through nodes
    def fit_poly_through_nodes(self, nodes, fit_cond_max_dst=10.0, fit_cond_max_mse=0.001):
        # linear integration to get the t coordinate element-wise
        dx = nodes[1:, 0] - nodes[:-1, 0]
        dy = nodes[1:, 1] - nodes[:-1, 1]
        dst = np.sqrt(dx ** 2 + dy ** 2)
        dst = np.insert(np.cumsum(dst), 0, 0)
        path_length = dst.sum()
        t = dst/dst[-1]  # normalize

        # interpolate (linear)
        lin_spline_x = interpolate.interp1d(t, nodes[:, 0], kind='linear')
        lin_spline_y = interpolate.interp1d(t, nodes[:, 1], kind='linear')

        # search for the best fit (satisfies fit_cond_max_dst and fit_cond_max_mse)
        # increasing knots number one by one
        # assumption: evenly distributed knots
        for knots_num in range(2, nodes.shape[0]):

            # resample new knots
            t = np.linspace(0, 1, knots_num)
            knots_x = lin_spline_x(t)
            knots_y = lin_spline_y(t)
            # fit spline through new knots
            road_geometry = self.fit_poly_through_nodes_exact(np.array([knots_x, knots_y]).T)

            # compute fitting errors
            t = np.linspace(0, knots_num - 1, 4 * nodes.shape[0])
            x_sampled = self.cs_over_x(t)
            y_sampled = self.cs_over_y(t)
            line = geom.LineString(np.array([x_sampled, y_sampled]).T)
            mse = []
            for node in nodes:
                point = geom.Point(node[0], node[1])
                mse.append(line.distance(point))
            mse = np.array(mse)
            max_dst = np.max(mse)
            mse = mse.mean()/path_length
            if mse < fit_cond_max_mse and max_dst < fit_cond_max_dst:
                return road_geometry, (knots_num, knots_x, knots_y, mse, max_dst)

        raise Exception('Could not fit polynomial')

    # deletes nodes that are outside the GPS track
    def trim_nodes(self, nodes, road_data, coords_begin, coords_end):
        nodes = np.array(nodes)
        # get coordinates of the 1st and the last GPS points
        easting_begin, northing_begin, _, _ = utm.from_latlon(coords_begin[0], coords_begin[1])
        easting_end, northing_end, _, _ = utm.from_latlon(coords_end[0], coords_end[1])
        coords_begin_end = [np.array([easting_begin, northing_begin]), np.array([easting_end, northing_end])]
        # storage for indexes for trimming
        trim_road_data = [None, None]
        trim_nodes = [None, None]

        for coords_idx, coords in enumerate(coords_begin_end):
            # find the index of a closest node to the 1st and the last GPS points
            distances_to_coords = np.power(nodes - coords, 2).sum(1)
            distances_to_coords = np.argsort(distances_to_coords)
            # check lines combined from adjacent points
            idxs_lst = [[distances_to_coords[0], distances_to_coords[0] + 1],
                        [distances_to_coords[0] - 1, distances_to_coords[0]]]
            projection = None
            for idxs in idxs_lst:
                projection = self.check_projection(nodes, idxs, coords)

                # check end conditions
                # - non useful nodes
                # - should we delete the 1st or last ways (if we trim after)
                if np.any(projection):
                    if not coords_idx:  # begin
                        trim_nodes[0] = min(idxs)
                        nodes[trim_nodes[0]] = projection
                        trim_road_data[0] = None if trim_nodes[0] < road_data[0]['s_coord'] else 1
                    else:  # end
                        trim_nodes[1] = max(idxs)
                        nodes[trim_nodes[1]] = projection
                        if len(road_data) > 1:
                            trim_road_data[1] = None if trim_nodes[1] > road_data[-2]['s_coord'] else -1
                    break
            if not np.any(projection):
                raise Exception('Could not project end coordinates')

        return nodes[trim_nodes[0]:trim_nodes[1] + 1], road_data[trim_road_data[0]:trim_road_data[1]]


    def get_dst(self, a, b, axis=None):
        return np.sqrt(((a - b) ** 2).sum(axis))


    def check_projection(self, nodes, idxs, point):
        # if the point not within nodes -> no projection is possible
        for idx in idxs:
            if idx < 1 or idx == nodes.shape[0]:
                return None

        line = nodes[idxs]
        projection = self.project_coords_to_line(line, point)
        # distance between the line points
        dst_between = self.get_dst(line[0], line[1])
        # distance between the line points and the projected points
        dst_projection = self.get_dst(line[0], projection) + self.get_dst(line[1], projection)

        # distances are equal if the projected point is in between line points
        if abs(dst_between - dst_projection) < 1e-8:
            return projection
        else:
            return None  # otherwise projection is on another line


    def project_coords_to_line(self, line, coords):
        # projection formula: proj(coords) = s0 + dst * (s1 - s0) / | s1 - s0 |
        # where s0, s1 line start/end coords; dst  - distance from the point to the line

        # ensure the data format is np.array float
        s0 = np.asarray(line[0], dtype=float)
        s1 = np.asarray(line[1], dtype=float)
        coords = np.asarray(coords, dtype=float)

        # implementation projection formula
        n = s1 - s0
        n /= np.linalg.norm(n, 2)
        projection = s0 + n * np.dot(coords - s0, n)

        return projection


    def equal_dicts(self, d1, d2, ignore_keys):
        ignored = set(ignore_keys)
        for k1, v1 in d1.items():
            if k1 not in ignored and (k1 not in d2 or d2[k1] != v1):
                return False
        for k2, v2 in d2.items():
            if k2 not in ignored and k2 not in d1:
                return False
        return True


    def try_int(self, string):
        try:
            return int(string)
        except ValueError:
            return string


    def tags_to_list(self, tags):
        if tags:
            listed_tags = tags.split('|')
            listed_tags = ['none' if x == '' else x for x in listed_tags]
            listed_tags = [item.split(';') for item in listed_tags]
            return listed_tags
        else:
            return None


    def get_poly_coefs_from_road_segment_dict(self, road_geometry_segment):
        coeffs_x = np.array([road_geometry_segment['x'], road_geometry_segment['bU'], road_geometry_segment['cU'], road_geometry_segment['dU']])
        coeffs_y = np.array([road_geometry_segment['y'], road_geometry_segment['bV'], road_geometry_segment['cV'], road_geometry_segment['dV']])
        return coeffs_x, coeffs_y


    def update_s_coord_road_segments_linear(self, road_data, nodes):
        distances = np.sqrt(((nodes[1:] - nodes[:-1]) ** 2).sum(1))
        begin_idx = 0
        for road_segment in road_data:
            end_idx = road_segment['s_coord']
            road_segment['s_coord'] = distances[:begin_idx].sum()
            road_segment['s_length'] = distances[begin_idx:end_idx].sum()
            begin_idx = end_idx
        return road_data


    def fit_poly_through_nodes_exact(self, nodes):
        distances = np.sqrt(((nodes[1:] - nodes[:-1]) ** 2).sum(1))
        s_coords = np.insert(np.cumsum(distances), 0, 0)
        nodes = np.asarray(nodes)
        x = nodes[:, 0]
        y = nodes[:, 1]
        knots = nodes.shape[0]
        t = np.arange(knots)
        bc = 'natural'
        self.cs_over_x = CubicSpline(t, x, bc_type=bc)
        self.cs_over_y = CubicSpline(t, y, bc_type=bc)

        # debug array with coefs
        polyfit_coefs = np.zeros((knots - 1, 8))
        polyfit_coefs[:, 0:4] = self.cs_over_x.c[::-1, :].T
        polyfit_coefs[:, -4:] = self.cs_over_y.c[::-1, :].T

        poly_segments = []
        for idx, poly_segment_coefs in enumerate(polyfit_coefs):
            geometry_dict = {'s': s_coords[idx], 'x': poly_segment_coefs[0], 'y': poly_segment_coefs[4], 'length': distances[idx],
                             'aU': 0, 'bU': poly_segment_coefs[1], 'cU': poly_segment_coefs[2], 'dU': poly_segment_coefs[3],
                             'aV': 0, 'bV': poly_segment_coefs[5], 'cV': poly_segment_coefs[6], 'dV': poly_segment_coefs[7]}
            poly_segments.append(geometry_dict)

        return poly_segments

    # returns: 0 - not connected
    #          1 - way 2 follows way 1
    #          2 - way 1 follows way 1
    def check_ways_connection(self, way_1, way_2):
        if way_2.nodes[0].id == way_1.nodes[-1].id:
            return 1
        elif way_1.nodes[0].id == way_2.nodes[-1].id:
            return 2
        else:
            return 0


    # Provides information about the country/region per very listed pair of WGS coordinates
    # output - the list of dictonaries {'country_code': country_code, 'country': country, 'region': region, 'sub_region': sub_region, 'error_code': error_code}
    # error codes:
    #   0 - no error (full output data is available)
    #   1 - could not get the country code from geolocator (output data is not available)
    #   2 - could not find the country in the database (output data is available partially)
    # if cannot initialize the geoloacator or cannot find/load ISO-3166 database (csv) will rise exception
    def get_region_name_from_wgs(self, coords_list,  # the list of tuples (latitude, longitude)
                             user_agent='TNO',  # describes who makes the request, can be anything
                             verbose=False,  # allows to print the input and output data
                             region_idx=5,  # region column number in the CSV database
                             sub_region_idx=6):  # sub_region column number in the CSV database
        output = []
        error_code = 0
        geolocator = Nominatim(user_agent=user_agent)
        csv_region_database = list(csv.reader(open(self.region_database, "r")))

        # get region names from the coordinates list one by one
        for coords in coords_list:
            country, country_code, region, sub_region = None, None, None, None
            try:
                location = geolocator.reverse(coords, language='en', addressdetails=True)
                country_code = location.raw['address']['country_code'].upper()
                country = location.raw['address']['country']
                # try to search the code in the databse
                for row in csv_region_database:
                    if country_code in row:
                        error_code = 0
                        region = row[region_idx]
                        sub_region = row[sub_region_idx]
                        break
                    else:
                        error_code = 2
            except:
                error_code = 1
            if verbose:
                if error_code == 1:
                    print('lat, lon:', coords, 'error: could not get the country code from geolocator')
                elif error_code == 2:
                    print('lat, lon:', coords, '-> country_code:', country_code, 'country:', country, 'error: could not find the country in the database')
                else:
                    print('lat, lon:', coords, '-> country_code:', country_code, 'country:', country, 'region:', region, 'sub_region:', sub_region)
            region_dict = {'country_code': country_code, 'country': country, 'region': region, 'sub_region': sub_region, 'error_code': error_code}
            output.append(region_dict)
            return output
