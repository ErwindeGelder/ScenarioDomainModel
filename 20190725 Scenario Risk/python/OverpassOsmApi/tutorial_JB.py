import numpy as np
import matplotlib.pyplot as plt
import gpxpy.gpx
from OverpassOsmApi import OverpassOsmApi
import utm


# important knobs
gpx_file = './from_JB/Irschenberg.gpx'
max_dist_from_motorway = 30  # the distance to way from the ego vehicle, m, where the correct road is expected,
                             # should be within GPS accuracy in meters, 20 is a good value for the data in the example
fit_cond_max_dst = 10.0  # polynomial fitting condition: maximal distance from a OSM node to the fitted curve
fit_cond_max_mse = 0.001  # polynomial fitting condition: average distance from a OSM node to the fitted curve
                          # (normalized by the path lengh)
# less important knobs
overpass_servers = 'overpass_servers.txt'  # thee list of servers to retrieve
region_database = 'countries.csv' # the path to the CSV with database ISO-3166-Countries-with-Regional-Codes,
                # taken from: https://github.com/lukes/ISO-3166-Countries-with-Regional-Codes/blob/master/all/all.csv
buffered_arrea_bbox_margin = 800  # the size of a piece of map that is cashed, m (800 is optimal)


# read the GPS points to process further
gpx = gpxpy.parse(open(gpx_file))
coords = []
for track in gpx.tracks:
    for segment in track.segments:
        for point in segment.points:
            coords.append((point.latitude, point.longitude))

# initialize the API
osm = OverpassOsmApi(buffered_arrea_bbox_margin=buffered_arrea_bbox_margin,
                     max_dist_from_motorway=max_dist_from_motorway,
                     url_list=None,
                     url_list_file=overpass_servers,
                     region_database=region_database)

print('readed', len(coords), 'coords')

# get region name from the 1st coordinate
region = osm.get_region_name_from_wgs([coords[0]], verbose=False)
# get routes from WGS coords, routes are sorted by distance, no mre than 10 routes
routes = osm.get_routes_from_coords_list(coords)
# get nodes from the closest route as a joined array
nodes, road_data = osm.get_road_data_from_list_of_ways(routes[0])
# store for plotting, no needed for further processing
nodes_old = nodes
# trim route ends that exceed current coordinate ranges
nodes, road_data = osm.trim_nodes(nodes, road_data, coords[0], coords[-1])
# get s coordinates by integration. note integrates over linear interpolation over nodes
road_data = osm.update_s_coord_road_segments_linear(road_data, nodes)
# getting road geometry from nodes
road_geometry, fit_data = osm.fit_poly_through_nodes(nodes, fit_cond_max_dst=fit_cond_max_dst,
                                                            fit_cond_max_mse=fit_cond_max_mse)
# print results
print('\nregion')
print(region[0])

print('\nroad_data')
for idx, road_element in enumerate(road_data):
    print(idx, road_element)

print('\nroad_data lane turns')
for idx, road_element in enumerate(road_data):
    print(idx, 'lanes:', road_element['lanes'], road_element['turn_lanes'], 'length:', int(road_element['s_length']))

print('\nroad_geometry')
for rge_idx, road_geometry_element in enumerate(road_geometry):
    print(rge_idx, road_geometry_element)

# evaluation of fitted polynomial elements via OSM nodes
# convert wgs to utm for plotting
utm_x, utm_y = [], []
for coord in coords:
    utm_x_, utm_y_, _, _ = utm.from_latlon(coord[0], coord[1])
    utm_x.append(utm_x_)
    utm_y.append(utm_y_)

# evaluate the new road curve
total_t = 100
t = np.linspace(0, 1, total_t)
poly_t_ev = np.array([np.ones(total_t), t, t ** 2, t ** 3])
for road_geometry_segment in road_geometry:
    coeffs_x, coeffs_y = osm.get_poly_coefs_from_road_segment_dict(road_geometry_segment)
    x_ev = np.dot(coeffs_x,  poly_t_ev)
    y_ev = np.dot(coeffs_y, poly_t_ev)
    plt.plot(x_ev, y_ev)

# plot knots, GPS points, OSM nodes, etc
(knots_num, knots_x, knots_y, MSE, max_distance) = fit_data  # store for plotting, no needed for further processing
plt.plot(knots_x, knots_y, 'om', label='knots (cubic poly)', markersize=10, fillstyle='none')
plt.plot(utm_x, utm_y, '*b', markersize=8, label='GPS points')
plt.plot(nodes_old[:, 0], nodes_old[:, 1], 'r.', label='OSM nodes', markersize=10)
plt.plot(nodes[:, 0], nodes[:, 1], 'k.', label='OSM trimmed nodes')
plt.title('Fitted polynomial elements via OSM nodes (approx), \nknots: ' + str(knots_num) + ', MSE: % .6f' % MSE + ', max: % .2f' % max_distance)
plt.axis('equal')
plt.legend()
plt.grid()
plt.show()

# done 1. clip nodes relative to coords. now they start end with the 1st and last nodes of the wway, but should be closest nodes to gps points
# done 2. reduce poly order. now knots# = nodes# , but should be optimal, e.g. 3-5 in this example
# skip 3. (update_s_coord_road_segments_poly) integration error because of poor poly fitting
# skip 4. add functionality: zero center the curve and rotate to 0 deg