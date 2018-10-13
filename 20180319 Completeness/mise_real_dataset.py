import numpy as np
import pickle
import matplotlib.pyplot as plt
from fastkde import KDE
import os
from tqdm import tqdm_notebook as tqdm
import h5py
import psutil
from time import time

# Open the dataset
with open(os.path.join('pickles', 'df.p'), 'rb') as f:
    dfs, scaling = pickle.load(f)
scaling = scaling.T  # [time vstart vend]
scaling = scaling[scaling[:, 2] > 0, :]  # Remove full stops
scaling[:, 1] = scaling[:, 1] - scaling[:, 2]  # Now it becomes: [time deltav vend] (less correlation)
scaling[:, 0] = scaling[:, 1] / scaling[:, 0]  # Now it becomes: [deceleration deltav vend] (better behaved)
std_scaling = np.std(scaling, axis=0)
mean_scaling = np.mean(scaling, axis=0)
scaling = (scaling - mean_scaling) / std_scaling

# Parameters
nmin = 600
nsteps = 30
nmax = len(scaling)
xmin = -3
xmax = 7
nx = 101
nrepeat = 10
seed = 0
d = scaling.shape[1]
overwrite = False
max_change_bw = 0.25

# Initialize some stuff
nn = np.round(np.logspace(np.log10(nmin), np.log10(nmax), nsteps)).astype(np.int)
bw = np.zeros((nrepeat, len(nn)))
mise_var = np.zeros_like(bw)
mise_bias = np.zeros_like(bw)
mise = np.zeros_like(bw)
x = np.linspace(xmin, xmax, nx)
xx = np.transpose(np.meshgrid(*[x for _ in range(d)]), np.concatenate((np.arange(1, d+1), [0])))
dx = np.mean(np.gradient(x))

np.random.seed(seed)
data = scaling.copy()
filename = os.path.join('hdf5', 'mise_real_dataset.hdf5')
if overwrite or not os.path.exists(filename):
    for repeat in tqdm(range(nrepeat)):
        np.random.shuffle(data)
        kde = KDE(data=data)
        kde.set_score_samples(xx)

        # Evaluate MISE for different values of n
        starttime = time()
        for i, n in enumerate(nn):
            print("{:8.1f} Step {:d} from {:d}, n={:d}/{:d}".format(time()-starttime, i+1, len(nn), n, nmax))
            # Compute the bandwidth only if this is the first time
            kde.set_n(n)
            print("{:8.1f} Compute bandwidth".format(time() - starttime), end="")
            if i == 0:
                kde.compute_bw()
            else:
                kde.compute_bw(min_bw=kde.bw*(1-max_change_bw), max_bw=kde.bw*(1+max_change_bw))
            bw[repeat, i] = kde.bw
            print(" --> {:.6f}".format(kde.bw))

            # Compute the pdf
            print("{:8.1f} Compute pdf".format(time() - starttime))
            pdf = kde.score_samples()

            # Compute integral of Laplacian
            print("{:8.1f} Compute Laplacian".format(time() - starttime))
            laplacian = kde.laplacian()
            # print("{:8.1f} Compute integral Laplacian".format(time() - starttime))
            integral = laplacian ** 2
            for _ in range(d):
                integral = np.trapz(integral, x)

            # Estimate MISE
            # print("{:8.1f} Estimate MISE".format(time() - starttime))
            mise_var[repeat, i] = kde.muk / (n * bw[repeat, i] ** d)
            mise_bias[repeat, i] = bw[repeat, i] ** 4 / 4 * integral
            mise[repeat, i] = mise_var[repeat, i] + mise_bias[repeat, i]
            print("Mise = {:.6e} + {:.6e} = {:.6e}".format(mise_var[repeat, i], mise_bias[repeat, i], mise[repeat, i]))

            print("Estimated total time: {:.1f} s".format((time() - starttime) * np.sum(nn) / np.sum(nn[:i+1])))

        process = psutil.Process(os.getpid())
        print(process.memory_info().rss)

    # Write data to file
    with h5py.File(filename, "w") as f:
        f.create_dataset("mise_var", data=mise_var)
        f.create_dataset("mise_bias", data=mise_bias)
        f.create_dataset("mise", data=mise)
        f.create_dataset("bw", data=bw)
else:
    with h5py.File(filename, "r") as f:
        mise_var = f["mise_var"][:]
        mise_bias = f["mise_bias"][:]
        mise = f["mise"][:]
        bw = f["bw"][:]

for m in mise:
    plt.loglog(nn, m, color=[.5, .5, 1])
plt.loglog(nn, np.mean(mise, axis=0), 'r-', lw=3, label="Mean")
plt.xlabel("Number of samples")
plt.ylabel("MISE")
plt.legend()
plt.grid(True)
plt.show()

f, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.loglog(nn, mise[0], lw=5)
ax.set_xlabel("Number of samples")
ax.set_ylabel("MISE")
ax.set_xlim([nn[0], nn[-1]])
ax.grid(True)
print("Power for estimated MISE: {:.2f}".format(np.polyfit(np.log(nn), np.log(mise[0]), 1)[0]))
plt.show()
