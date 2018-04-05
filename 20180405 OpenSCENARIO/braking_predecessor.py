from lxml import builder
from lxml import etree
import datetime


# Parameters
vinit = 50/3.6  # [m/s] Initial speed of both vehicles
dinit = vinit * 1.5  # [s] Initial distance between both vehicles
t0 = 5  # [s] Time before braking
tbraking = 5  # [s] Duration of the braking
tend = 20  # [s] End time of the scenario
vend = 10/3.6  # [m/s] End speed of predecessor
ego_name = "Ego"
target_name = "Predecessor"

# Create header stuff
e = builder.ElementMaker()
doc = e.OpenSCENARIO(e.FileHeader(revMajor="0", revMinor="1",
                                  date=datetime.datetime.today().strftime('%Y-%m-%dT%H:%M:%S'),
                                  description="Braking predecessor",
                                  author="Erwin de Gelder"))
doc.append(e.Entities(e.Object(name=ego_name),
                      e.Object(name=target_name)))

# Define initial state
ego_init = e.Private(e.Action(e.Longitudinal(e.Speed(e.Dynamics(shape="step"),
                                                     e.Target(e.Absolute(value="{:.1f}".format(vinit)))))),
                     e.Action(e.Position(e.World(x="0", y="0", z="0",
                                                 h="0", p="0", r="0"))),
                     object=ego_name)
target_init = e.Private(e.Action(e.Position(e.World(x="{:.1f}".format(dinit),
                                                    y="0", z="0",
                                                    h="0", p="0", r="0"))),
                        object=target_name)
doc.append(e.Storyboard(e.Init(e.Actions(ego_init,
                                         target_init))))
storyboard = doc[-1]

# Define maneuver of predecessor
cruising1 = e.Event(e.Action(e.Private(e.Longitudinal(e.Speed(e.Dynamics(shape="linear", time="{:.1f}".format(t0)),
                                                              e.Target(e.Absolute(value="{:.1f}".format(vinit))))))),
                    e.StartConditions(e.ConditionGroup(e.Condition(e.ByState(e.AtStart(type="story",
                                                                                       name="MyStory"))))),
                    name="Cruising1", priority="overwrite")
braking = e.Event(e.Action(e.Private(e.Longitudinal(e.Speed(e.Dynamics(shape="sinusoidal",
                                                                       time="{:.1f}".format(tbraking)),
                                                            e.Target(e.Absolute(value="{:.1f}".format(vend))))))),
                  e.StartConditions(e.ConditionGroup(e.Condition(e.ByState(e.AfterTermination(type="action",
                                                                                              name="Cruising1",
                                                                                              rule="end"))))),
                  name="Braking", priority="overwrite")
cruising2 = e.Event(e.Action(e.Private(e.Longitudinal(e.Speed(e.Dynamics(shape="linear",
                                                                         time="{:1f}".format(tend-t0-tbraking)),
                                                              e.Target(e.Absolute(value="{:.1f}".format(vend))))))),
                    e.StartConditions(e.ConditionGroup(e.Condition(e.ByState(e.AfterTermination(type="action",
                                                                                                name="Braking",
                                                                                                rule="end"))))),
                    name="Cruising2", priority="overwrite")
storyboard.append(e.Story(e.Act(e.Sequence(e.Actors(e.Entity(name="$owner")),
                                           e.Maneuver(cruising1, braking, cruising2,
                                                      name="BrakingPredecessorManeuver"),
                                           name="MySequence"),
                                name="MyAct"),
                          name="MyStory", owner=target_name))

doc = etree.tostring(doc, xml_declaration=True, encoding='utf-8', pretty_print=True)
with open("braking_predecessor.xosc", "wb") as f:
    f.write(doc)
print(doc.decode('ascii'))
