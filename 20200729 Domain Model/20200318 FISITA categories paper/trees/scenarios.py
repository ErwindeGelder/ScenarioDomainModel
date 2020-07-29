""" Create the figures with the scenarios. """

from glob import glob
import os
from shutil import copyfile
from subprocess import call
from jinja2 import Environment


DOCUMENTS = [("cutin", ("\\scentags{ >Ego; Ego lateral activity, Going straight; " +
                        "Vehicle longitudinal activity, Driving forward|" +
                        ">Actor; Carriageway user type, Vehicle; Initial state, " +
                        "Direction/Same as ego; Lead vehicle, Appearing/Cutting in; " +
                        "Vehicle lateral activity, Changing lane; " +
                        "Vehicle longitudinal activity, Driving forward| " +
                        "Road layout, Straight, Merge}")),
             ("oncoming_turning", ("\\scentags{ >Ego; Vehicle lateral activity, Going straight; " +
                                   "Vehicle longitudinal activity, Driving forward|" +
                                   ">Actor; Carriageway user type, Vehicle; Initial state, " +
                                   "Direction/Oncoming, Lateral position/Right of ego, " +
                                   "Long.\\ position/In front of ego; Vehicle lateral activity, " +
                                   "Turning/Right|" +
                                   "Road layout, Junction, Traffic light }")),
             ("straight", ("\\scentags{Longitudinal activity, Driving forward|" +
                           "Lateral activity, Following lane|" +
                           "Road layour, Straight }"))]


with open('template_scenario.tex') as tpl:
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
