""" Create the figures with the trees. """

from glob import glob
import os
from shutil import copyfile
from subprocess import call
from jinja2 import Environment


DOCUMENTS = [("actor_type", ("\\tree[\\node[tag] at (-3\\hwidth, -0.3em) {Vehicle};" +
                             "\\draw[draw=black, line width=0.5mm] (-4\\hwidth+0.7em, -1em) " +
                             "rectangle (2\\hwidth+.8em,-10.8em);]" +
                             "{Carriageway user type}{Category M, Pas.\\ car (M1),Minibus (M2), " +
                             "Bus (M3); Category N, LCV (N1), LGV (N2 N3); Category L, " +
                             "Moped (L1), Motorcycle (L3); VRU, Pedestrian, Cyclist, Other}")),
             ("lat_activity", ("\\tree{Vehicle lateral activity}{Going straight; Changing lane, " +
                               "Left, Right; Turning, Left, Right; Swerving, Left, Right}")),
             ("lon_activity", ("\\tree{Vehicle longitudinal activity}{Reversing; Standing still; " +
                               "Driving forward, Decelerating, Cruising, Accelerating}")),
             ("ped_activity", ("\\tree{Pedestrian activity}{Walking, Straight, Turning left, " +
                               "Turning right; Stopping; Standing still}")),
             ("cyc_lat_activity", ("\\tree{Cyclist lateral activity}{Going straight; Turning, " +
                                   "Left, Right; Swerving, Left, Right}")),
             ("cyc_lon_activity", ("\\tree{Cyclist longitudinal activity}{Riding forward, " +
                                   "Decelerating, Cruising, Accelerating; Stopping; " +
                                   "Standing still}")),
             ("initial_state", ("\\tree{Initial state}{Direction, Same as ego, Oncoming, " +
                                "Crossing; Dynamics, Moving, Standing still; Lateral position," +
                                " Same lane, Left of ego, Right of ego; Long.\\ position, " +
                                "In front of ego,Side of ego,Rear of ego}")),
             ("lead_vehicle", ("\\tree{Lead vehicle}{Appearing, Cutting in, Gap closing; " +
                               "Disappearing, Cutting out, Gap opening; Following}")),
             ("animal", ("\\tree{Animal}{Position, On ego path, On road, Next to road; Dynamics, " +
                         "Moving, Stationary}")),
             ("road_type", ("\\tree{Road type}{Principal road, Motorway, Trunk, Primary, " +
                            "Secondary, Tertiary, Unclassified, Residential, Service;  Link, " +
                            "Motorway link, Trunk link, Primary link, Secondary link, " +
                            "Tertiary link, Sliproad; Pavement, Footway, Cyclist path}")),
             ("road_layout", ("\\tree{Road layout}{Straight, Merge, Entrance, Exit, Other; " +
                              "Curved, Merge, Entrance, Exit, Other; Junction, Traffic light, " +
                              "No traffic light, Roundabout; Ped.\\ crossing, Traffic light, " +
                              "No traffic light}")),
             ("static_object", ("\\tree{Static object}{On path, Passable, Impassable; " +
                                "View blocking; Other}")),
             ("traffic_light", "\\tree{Traffic light}{Red; Amber; Green; N.A.}"),
             ("weather", ("\\tree{Weather}{No precipitation; Rain, Light, Moderate, Heavy; " +
                          "Suspension, Mist, Fog, Haze; Snow, Light, Moderate, Heavy}")),
             ("lighting", ("\\tree{Lighting}{Daytime, Clear sky, Cloudy, Overcast; Twilight, " +
                           "Dawn, Dusk; Dark, Street lights, No street lights; Glare, Sun, " +
                           "Oncom.\\ traffic}"))]


with open('template.tex') as tpl:
    template = tpl.read()


def write_file(**kwargs):
    string = Environment().from_string(template).render(**kwargs)
    with open('tree.tex', 'w') as file:
        file.write(string)


if __name__ == "__main__":
    for name, content in DOCUMENTS:
        write_file(document=content)
        call("pdflatex.exe tree.tex -synctex=1 -interaction=nonstopmode")
        copyfile("tree.pdf", os.path.join("..", "figures", "{:s}.pdf".format(name)))
    for remove_file in glob("tree.*"):
        os.remove(remove_file)
