# Scenario Domain Model

This repository contains a Python implementation of an ontology for scenarios 
for the assessment of automated vehicles. Whenever one uses this repository, 
please refer to the following publication:

E. de Gelder, J.-P. Paardekooper, A. Khabbaz Saberi, H. Elrofai, O. Op den Camp, 
J. Ploeg, L. Friedmann, and B. De Schutter, "Ontology for Scenarios for the 
Assessment of Automated Vehicles", 2019. *In preparation.*

More details on the terms and definitions can also be found in the 
aforementioned publication. 

We will soon add more information.

# Running the code

Every file in the `examples` folder can be executed with Python provided that 
Python can find the folder `domain_model`. To make sure that Python can find the 
folder `domain_model`, its parent folder needs to be added to the so-called 
PYTHONPATH. This can be done as follows 
([here](https://bic-berkeley.github.io/psych-214-fall-2016/using_pythonpath.html) 
you can find more information):

- On Windows, suppose that the whole repository is stored locally in the folder 
`C:\ScenarioDomainModel\`. In that case, the folder `C:\ScenarioDomainModel\` 
should be added to the Python path. Go to the Control Panel, search for 
"Environment Variables" and click the result and make sure you are in Admin 
mode. A panel appears with two textfields ("User variables for XXX" and 
"System variables"). Under "system variables", there is a variable names 
`PYTHONPATH`. Click on it and click "Edit...". A new panel appears in which you 
can click on "New" and type your path (e.g., `C:\ScenarioDomainModel\`). Finish 
by clicking multiple times on "OK".

- On Linux, suppose that the whole repository is stored locally in the folder 
`~/ScenarioDomainModel/`. Add the line 
`export PYTHONPATH=~/ScenarioDomainModel`.
