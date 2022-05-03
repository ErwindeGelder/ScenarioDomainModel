# Scenario Domain Model

This repository contains a Python implementation of an ontology for scenarios 
for the assessment of automated vehicles. Whenever one uses this repository, 
please refer to the following publication:

E. de Gelder, J.-P. Paardekooper, A. Khabbaz Saberi, H. Elrofai, O. Op den Camp, 
S. Kraines, J. Ploeg, and B. De Schutter, "Towards an Ontology for Scenario 
Definition for the Assessment of Automated Vehicles: An Object-Oriented 
Framework", [*IEEE Transactions on Intelligent Vehicles, Early access*](https://doi.org/10.1109/TIV.2022.3144803), 2022.
Note: a *public* preprint is available at [arXiv](https://arxiv.org/abs/2001.11507).

More details on the terms and definitions can also be found in the 
aforementioned publication. 

The best way to learn more about this code is to have a look at the tutorials:

1. [Instatiating a *scenario category*](./Tutorial%201%20Scenario%20category.ipynb)
2. [Instatiating a *scenario*](./Tutorial%202%20Scenario.ipynb)
3. [Creating a *scenario* from data](./Tutorial%203%20Scenario%20from%20data.ipynb)
4. [Scenario database](./Tutorial%204%20Scenario%20database.ipynb)
5. [Instantiating a *scenario category* including I2V communication](./Tutorial%205%20Scenario%20category%20including%20I2V%20communication.ipynb)

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

# Installing the package

The domain model from this repository can be installed as a Python package and quickly reused in other projects. Go to the package directory and type in a console:

```sh
pip3 install .
```

To import the package in another project, use the standard importing mechanisms:

```python
import domain_model
# Or
from domain_model import Actor
```
