import time
import numpy as np
from simulation import SimulationLeadBraking


simulator = SimulationLeadBraking()
v0 = 30
amean = 2.3 # 2.3  # 2.3
dv = 20
tstart = time.time()
np.random.seed(1)
print(simulator.get_probability((v0, amean, dv), plot=True))
print("Elapsed time: {:.3f}".format(time.time()-tstart))
