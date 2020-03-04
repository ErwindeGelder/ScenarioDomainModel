""" Test the MISE method for a univariate distribution.

Creation date: 2018 10 09
Author(s): Erwin de Gelder

Modifications:
2020 03 04 Make PEP8 compliant and use KDE from own stats library.
"""

import os
import numpy as np
from matplotlib2tikz import save
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
from stats import KDE, GaussianMixture

# Parameters
NMAX = 5000
NMIN = 100
NSTEPS = 50
SEED = 1
OVERWRITE = False
NREPEAT = 200
NPDF = 101
MU = np.array([-1, 1])
SIGMA = np.array([0.5, 0.3])
XLIM = [-3, 3]
MAX_CHANGE_BW = 0.25
FILENAME = os.path.join("hdf5", "mise_univariate2.hdf5")

# Generate datapoints
np.random.seed(SEED)
GM = GaussianMixture(MU, SIGMA)
X = GM.generate_samples(NMAX * NREPEAT).reshape((NREPEAT, NMAX))

# This we need for the next step
(XPDF,), YPDF = GM.pdf(minx=[XLIM[0]], maxx=[XLIM[1]], npoints=NPDF)
NN = np.round(np.logspace(np.log10(NMIN), np.log10(NMAX), NSTEPS)).astype(np.int)
DELTAX = np.mean(np.gradient(XPDF))
MUK = 1 / (2 * np.sqrt(np.pi))

if OVERWRITE or not os.path.exists(FILENAME):
    REAL_MISE = np.zeros(len(NN))
    EST_MISE = np.zeros((len(NN), NREPEAT))

    KDE_ESTIMATED = np.zeros((len(NN), NREPEAT, len(XPDF)))
    LAPLACIANS = np.array(np.zeros_like(KDE_ESTIMATED))
    BANDWIDTH = np.zeros(len(NN))

    # We use the first set of datapoints (i.e., X[0]) for determining the bandwidth
    # This has quite some influence on the remainder of the procedure...
    I_BW = 0

    # For now, just set the data for all the kdes
    print("Initialize all KDEs")
    KDES = [KDE(data=x) for x in X]
    for kde in KDES:
        kde.set_score_samples(XPDF)

    print("Compute the bandwidth for varying number of datapoints")
    for i, n in enumerate(tqdm(NN)):
        # Compute the optimal bandwidth using 1-leave-out
        KDES[I_BW].set_n(n)
        if i == 0:
            KDES[I_BW].compute_bandwidth()
        else:
            KDES[I_BW].compute_bandwidth(min_bw=KDES[I_BW].bandwidth * (1 - MAX_CHANGE_BW),
                                         max_bw=KDES[I_BW].bandwidth * (1 + MAX_CHANGE_BW))
        BANDWIDTH[i] = KDES[I_BW].bandwidth

    print("Estimate the pdfs and Laplacians")
    for j in tqdm(range(NREPEAT)):
        for i, n in enumerate(NN):
            # Compute all the nrepeats pdfs. We need to do this many times in order to determine
            # the real MISE
            KDES[j].set_n(n)
            KDES[j].set_bandwidth(BANDWIDTH[i])
            KDE_ESTIMATED[i, j] = KDES[j].score_samples()
            LAPLACIANS[i, j] = KDES[j].laplacian()

    print("Estimate the MISE")
    for i, (y, laplacian, h, n) in enumerate(zip(KDE_ESTIMATED, LAPLACIANS, BANDWIDTH, NN)):
        REAL_MISE[i] = np.trapz(np.mean((y - YPDF) ** 2, axis=0), XPDF)
        integral_laplacian = np.trapz(laplacian ** 2, XPDF)
        EST_MISE[i] = MUK / (n * h) + integral_laplacian * h ** 4 / 4

    # Write results to hdf5 file
    with h5py.File(FILENAME, "w") as f:
        f.create_dataset("real_mise", data=REAL_MISE)
        f.create_dataset("est_mise", data=EST_MISE)
        f.create_dataset("bw", data=BANDWIDTH)
else:
    with h5py.File(FILENAME, "r") as f:
        REAL_MISE = f["real_mise"][:]
        EST_MISE = f["est_mise"][:]
        BANDWIDTH = f["bw"][:]

# Compute power by which the MISE is decreasing
print("Power for real MISE:      {:.2f}".format(np.polyfit(np.log(NN), np.log(REAL_MISE), 1)[0]))
print("Power for estimated MISE: {:.2f}".
      format(np.polyfit(np.log(NN), np.log(EST_MISE[:, 0]), 1)[0]))

# Plot the real MISE and the estimated MISE and export plot to tikz
_, AXES = plt.subplots(1, 1, figsize=(7, 5))
AXES.loglog(NN, REAL_MISE, label="Real MISE", lw=5)
AXES.loglog(NN, EST_MISE[:, 0], label="Estimated MISE", lw=5)
AXES.set_xlabel("Number of samples")
AXES.set_ylabel("MISE")
AXES.legend()
AXES.grid(True)
AXES.set_xlim([np.min(NN), np.max(NN)])
# save(os.path.join('..', '20180924 Completeness paper', 'figures', 'mise_example.tikz'),
#      figureheight='\\figureheight', figurewidth='\\figurewidth')

# Plot again the real MISE and the estimated MISE and export plot to tikz
_, AXES = plt.subplots(1, 1, figsize=(7, 5))
EST_MEAN = np.mean(EST_MISE, axis=1)
EST_STD = np.std(EST_MISE, axis=1)
AXES.loglog(NN, REAL_MISE, label="Real MISE", lw=5, color=[0, 0, 0])
PLOT = AXES.loglog(NN, EST_MISE[:, 0], '--', label="Estimated MISE", lw=5, color=[.5, .5, .5])
# p = ax.loglog(nn, est_mean, label="Estimated MISE", lw=5, color=[.5, .5, .5])
AXES.fill_between(NN, EST_MEAN - 3 * EST_STD, EST_MEAN + 3 * EST_STD, color=[.8, .8, .8])
# ax.loglog(nn, est_mean-2*est_std, '.', color=p[0].get_color(), lw=5)
AXES.set_xlabel("Number of samples")
AXES.set_ylabel("MISE")
AXES.grid(True)
AXES.set_xlim([np.min(NN), np.max(NN)])
save(os.path.join('..', '20180924 Completeness paper', 'figures', 'mise_example.tikz'),
     figureheight='\\figureheight', figurewidth='\\figurewidth')

# Plot the bandwidth for varying n
_, AXES = plt.subplots(1, 1, figsize=(7, 5))
AXES.semilogx(NN, BANDWIDTH)
AXES.set_xlabel("Number of samples")
AXES.set_ylabel("Bandwidth")
AXES.grid(True)
AXES.set_xlim([np.min(NN), np.max(NN)])
plt.show()
