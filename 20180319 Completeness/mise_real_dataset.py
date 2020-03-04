""" Estimate the MISE for the parameters of the real data set.

Creation date: 2018 10 13
Author(s): Erwin de Gelder

Modifications:
2020 03 04 Make PEP8 compliant and use KDE from own stats library.
"""

import os
import pickle
from time import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
import h5py
import psutil
from matplotlib2tikz import save
from stats import KDE

# Open the dataset
with open(os.path.join('pickles', 'df.p'), 'rb') as f:
    _, SCALING = pickle.load(f)
SCALING = SCALING.T  # [time vstart vend]
SCALING = SCALING[SCALING[:, 2] > 0, :]  # Remove full stops

# Now it becomes: [time deltav vend] (less correlation)
SCALING[:, 1] = SCALING[:, 1] - SCALING[:, 2]
# Now it becomes: [deceleration deltav vend] (better behaved)
SCALING[:, 0] = SCALING[:, 1] / SCALING[:, 0]

SCALING = (SCALING - np.mean(SCALING, axis=0)) / np.std(SCALING, axis=0)

# Parameters
NMIN = 600
NSTEPS = 30
NMAX = len(SCALING)
XMIN = -3
XMAX = 7
N_X = 101
NREPEAT = 10
np.random.seed(0)
DIM = SCALING.shape[1]
DIMA = 1
DIMB = DIM - DIMA
OVERWRITE = False
MAX_CHANGE_BW = 0.5
FIGURES_FOLDER = os.path.join('..', '20180924 Completeness paper', 'figures')

# Initialize some stuff
NVEC = np.round(np.logspace(np.log10(NMIN), np.log10(NMAX), NSTEPS)).astype(np.int)
BANDWIDTHS = np.zeros((NREPEAT, len(NVEC)))
MISE_VAR = np.array(np.zeros_like(BANDWIDTHS))
MISE_BIAS = np.array(np.zeros_like(BANDWIDTHS))
MISE = np.array(np.zeros_like(BANDWIDTHS))
BWA = np.zeros((NREPEAT, len(NVEC)))
MISEA = np.array(np.zeros_like(BANDWIDTHS))
BWB = np.zeros((NREPEAT, len(NVEC)))
MISEB = np.array(np.zeros_like(BANDWIDTHS))
MISEAB = np.array(np.zeros_like(BANDWIDTHS))
XVEC = np.linspace(XMIN, XMAX, N_X)
XDATA = np.transpose(np.meshgrid(*[XVEC for _ in range(DIM)]),
                     np.concatenate((np.arange(1, DIM + 1), [0])))
XDATAA = np.transpose(np.meshgrid(*[XVEC for _ in range(DIMA)]),
                      np.concatenate((np.arange(1, DIMA + 1), [0])))
XDATAB = np.transpose(np.meshgrid(*[XVEC for _ in range(DIMB)]),
                      np.concatenate((np.arange(1, DIMB + 1), [0])))
DELTAX = np.mean(np.gradient(XVEC))

DATA = SCALING.copy()
FILENAME = os.path.join('hdf5', 'mise_real_dataset.hdf5')
if OVERWRITE or not os.path.exists(FILENAME):
    for repeat in tqdm(range(NREPEAT)):
        np.random.shuffle(DATA)
        kde = KDE(data=DATA)
        kde.set_score_samples(XDATA)

        # Evaluate MISE for different values of n
        starttime = time()
        for i, n in enumerate(NVEC):
            print("{:8.1f} Step {:d} for {:d}, n={:d}/{:d}".format(time()-starttime, i+1, len(NVEC),
                                                                   n, NMAX))
            # Compute the bandwidth only if this is the first time
            kde.set_n(n)
            print("{:8.1f} Compute bandwidth".format(time() - starttime), end="")
            if i == 0:
                kde.compute_bandwidth()
            else:
                kde.compute_bandwidth(min_bw=kde.bandwidth * (1 - MAX_CHANGE_BW),
                                      max_bw=kde.bandwidth * (1 + MAX_CHANGE_BW))
            BANDWIDTHS[repeat, i] = kde.bandwidth
            print(" --> {:.6f}".format(kde.bandwidth))

            # Compute the pdf
            print("{:8.1f} Compute pdf".format(time() - starttime))
            pdf = kde.score_samples()

            # Compute integral of Laplacian
            print("{:8.1f} Compute Laplacian".format(time() - starttime))
            laplacian = kde.laplacian()
            # print("{:8.1f} Compute integral Laplacian".format(time() - starttime))
            integral = laplacian ** 2
            for _ in range(DIM):
                integral = np.trapz(integral, XVEC)

            # Estimate MISE
            # print("{:8.1f} Estimate MISE".format(time() - starttime))
            MISE_VAR[repeat, i] = kde.constants.muk / (n * BANDWIDTHS[repeat, i] ** DIM)
            MISE_BIAS[repeat, i] = BANDWIDTHS[repeat, i] ** 4 / 4 * integral
            MISE[repeat, i] = MISE_VAR[repeat, i] + MISE_BIAS[repeat, i]
            print("Mise = {:.6e} + {:.6e} = {:.6e}".format(MISE_VAR[repeat, i],
                                                           MISE_BIAS[repeat, i], MISE[repeat, i]))

            print("Estimated total time: {:.1f} s".format((time()-starttime) * np.sum(NVEC) /
                                                          np.sum(NVEC[:i + 1])))

        process = psutil.Process(os.getpid())
        print(process.memory_info().rss)

    # Write data to file
    with h5py.File(FILENAME, "w") as f:
        f.create_dataset("mise_var", data=MISE_VAR)
        f.create_dataset("mise_bias", data=MISE_BIAS)
        f.create_dataset("mise", data=MISE)
        f.create_dataset("bw", data=BANDWIDTHS)
else:
    with h5py.File(FILENAME, "r") as f:
        MISE_VAR = f["mise_var"][:]
        MISE_BIAS = f["mise_bias"][:]
        MISE = f["mise"][:]
        BANDWIDTHS = f["bw"][:]

np.random.seed(0)
DATA = SCALING.copy()
FILENAME = os.path.join('hdf5', 'mise_real_dataset_split.hdf5')
if OVERWRITE or not os.path.exists(FILENAME):
    for repeat in tqdm(range(NREPEAT)):
        np.random.shuffle(DATA)
        kdea = KDE(data=DATA[:, :DIMA])
        kdeb = KDE(data=DATA[:, DIMA:])
        kdea.set_score_samples(XDATAA)
        kdeb.set_score_samples(XDATAB)

        # Evaluate MISE for different values of n
        starttime = time()
        for i, n in enumerate(NVEC):
            print("{:8.1f} Step {:d} for {:d}, n={:d}/{:d}".format(time()-starttime, i+1, len(NVEC),
                                                                   n, NMAX))
            # Compute the bandwidth only if this is the first time
            kdea.set_n(n)
            kdeb.set_n(n)
            print("{:8.1f} Compute bandwidth".format(time() - starttime), end="")
            if i == 0:
                kdea.compute_bandwidth()
                kdeb.compute_bandwidth()
            else:
                kdea.compute_bandwidth(min_bw=kdea.bandwidth * (1 - MAX_CHANGE_BW),
                                       max_bw=kdea.bandwidth * (1 + MAX_CHANGE_BW))
                kdeb.compute_bandwidth(min_bw=kdeb.bandwidth * (1 - MAX_CHANGE_BW),
                                       max_bw=kdeb.bandwidth * (1 + MAX_CHANGE_BW))
            BWA[repeat, i] = kdea.bandwidth
            BWB[repeat, i] = kdeb.bandwidth
            print(" --> {:.6f} and {:.6f}".format(kdea.bandwidth, kdeb.bandwidth))

            # Compute the pdf
            print("{:8.1f} Compute pdf".format(time() - starttime))
            pdfa = kdea.score_samples()
            pdfb = kdeb.score_samples()
            intpdfa = pdfa ** 2
            intpdfb = pdfb ** 2
            for _ in range(DIMA):
                intpdfa = np.trapz(intpdfa, XVEC)
            for _ in range(DIMB):
                intpdfb = np.trapz(intpdfb, XVEC)

            # Compute integral of Laplacian
            print("{:8.1f} Compute Laplacian".format(time() - starttime))
            laplaciana = kdea.laplacian()
            laplacianb = kdeb.laplacian()
            # print("{:8.1f} Compute integral Laplacian".format(time() - starttime))
            integrala = laplaciana ** 2
            integralb = laplacianb ** 2
            for _ in range(DIMA):
                integrala = np.trapz(integrala, XVEC)
            for _ in range(DIMB):
                integralb = np.trapz(integralb, XVEC)

            # Estimate MISE
            # print("{:8.1f} Estimate MISE".format(time() - starttime))
            MISEA[repeat, i] = (kdea.constants.muk / (n * BWA[repeat, i] ** DIMA) +
                                BWA[repeat, i] ** 4 / 4 * integrala)
            MISEB[repeat, i] = (kdeb.constants.muk / (n * BWB[repeat, i] ** DIMB) +
                                BWB[repeat, i] ** 4 / 4 * integralb)
            MISEAB[repeat, i] = (intpdfb * MISEA[repeat, i] + intpdfa * MISEB[repeat, i] +
                                 MISEA[repeat, i] * MISEB[repeat, i])
            print("Mise = {:.6e}".format(MISEAB[repeat, i]))
            print("{:02d} Estimated total time: {:.1f} s".format(repeat + 1,
                                                                 (time()-starttime)*np.sum(NVEC) /
                                                                 np.sum(NVEC[:i + 1])))

    # Write data to file
    with h5py.File(FILENAME, "w") as f:
        f.create_dataset("miseab", data=MISEAB)
        f.create_dataset("misea", data=MISEA)
        f.create_dataset("miseb", data=MISEB)
        f.create_dataset("bwa", data=BWA)
        f.create_dataset("bwb", data=BWB)
else:
    with h5py.File(FILENAME, "r") as f:
        MISEAB = f["miseab"][:]
        MISEA = f["misea"][:]
        MISEB = f["miseb"][:]
        BWA = f["bwa"][:]
        BWB = f["bwb"][:]

with open(os.path.join('pickles', 'df.p'), 'rb') as f:
    DFS, SCALING = pickle.load(f)
SCALING = SCALING.T  # [time vstart vend]
SCALING = SCALING[SCALING[:, 2] > 0, :]  # Remove full stops
SCALING[:, 1] = SCALING[:, 1] - SCALING[:, 2]  # Now it becomes: [time deltav vend]
SCALING[:, 0] = SCALING[:, 1] / SCALING[:, 0]  # Now it becomes: [deceleration deltav vend]
_, AXS = plt.subplots(3, 1, figsize=(7, 5))
for i, AXES in enumerate(AXS):
    AXES.hist(SCALING[:, i], bins=20, color=[.5, .5, .5], edgecolor=[0, 0, 0])
    xlim = AXES.get_xlim()
    AXES.set_xlim([0, xlim[1]])
AXS[0].set_xlabel('Average deceleration [m/s$^2$]')
AXS[1].set_xlabel('Speed difference [m/s]')
AXS[2].set_xlabel('End speed [m/s]')
for AXES in AXS:
    AXES.set_ylabel('Frequency')
plt.tight_layout()
save(os.path.join(FIGURES_FOLDER, 'histogram.tikz'),
     figureheight='\\figureheight', figurewidth='\\figurewidth',
     extra_axis_parameters=['axis x line*=bottom', 'axis y line*=left'])

LW = 1.75
_, AXES = plt.subplots(1, 1, figsize=(7, 5))
AXES.loglog(NVEC, BWA[0], '-', label="bwa", lw=LW, color=[.5, .5, .5])
AXES.loglog(NVEC, BWB[0], ':', label="bwb", lw=LW, color=[.5, .5, .5])
AXES.loglog(NVEC, BANDWIDTHS[0], '--', label="bwab", lw=LW, color=[0, 0, 0])
AXES.set_xlabel("Number of samples")
AXES.set_ylabel("Bandwidth")
AXES.set_xlim([np.min(NVEC), np.max(NVEC)])
save(os.path.join(FIGURES_FOLDER, 'bandwidth_real.tikz'),
     figureheight='\\figureheight', figurewidth='\\figurewidth',
     extra_axis_parameters=['xtick={600, 800, 1000, 1500, 2000, 2500}',
                            'xticklabels={600, 800, 1000, 1500, 2000, 2500}',
                            'axis x line*=bottom', 'axis y line*=left',
                            'every x tick/.style={black}', 'every y tick/.style={black}'])

_, AXES = plt.subplots(1, 1, figsize=(7, 5))
LOGFIT1 = np.polyfit(np.log(NVEC), np.log(MISE[0]), 1)
AXES.loglog(NVEC, np.exp(LOGFIT1[1]) * NVEC ** LOGFIT1[0], color=[0, 0, 0], lw=LW)
AXES.loglog(NVEC, MISE[0], lw=LW, color=[.5, .5, .5])
print("Approximation of MISE1: {:.3f}*n^{:.2f}".format(np.exp(LOGFIT1[1]), LOGFIT1[0]))
LOGFIT2 = np.polyfit(np.log(NVEC), np.log(MISEAB[0]), 1)
AXES.loglog(NVEC, np.exp(LOGFIT2[1]) * NVEC ** LOGFIT2[0], '--', color=[0, 0, 0], lw=LW)
AXES.loglog(NVEC, MISEAB[0], '--', lw=LW, color=[.5, .5, .5])
print("Approximation of MISE1: {:.3f}*n^{:.2f}".format(np.exp(LOGFIT2[1]), LOGFIT2[0]))
N2 = 1000
N1 = (np.exp(LOGFIT2[1] - LOGFIT1[1]) * N2 ** LOGFIT2[0]) ** (1 / LOGFIT1[0])
print("Number of points for MISE 1 to be equal to MISE 2 at n={:d}: {:.0f}".format(N2, N1))
AXES.set_xlim([np.min(NVEC), np.max(NVEC)])
AXES.set_xlabel("Number of samples")
AXES.set_ylabel("Measure for completeness")
save(os.path.join(FIGURES_FOLDER, 'mise_real.tikz'),
     figureheight='\\figureheight', figurewidth='\\figurewidth',
     extra_axis_parameters=['xtick={600, 800, 1000, 1500, 2000, 2500}',
                            'xticklabels={600, 800, 1000, 1500, 2000, 2500}',
                            'axis x line*=bottom', 'axis y line*=left',
                            'every x tick/.style={black}', 'every y tick/.style={black}'])

_, AXES = plt.subplots(1, 1, figsize=(7, 5))
for mise in MISE:
    AXES.loglog(NVEC, mise, color=[.5, .5, 1])
AXES.loglog(NVEC, MISE[0], lw=3, label="Estimated")
AXES.loglog(NVEC, np.mean(MISE, axis=0), 'r-', lw=3, label="Mean")
AXES.set_xlabel("Number of samples")
AXES.set_ylabel("MISE")
AXES.legend()
AXES.grid(True)

_, AXES = plt.subplots(1, 1, figsize=(7, 5))
for mise in MISEAB:
    plt.loglog(NVEC, mise, color=[.5, .5, 1])
AXES.loglog(NVEC, MISEAB[0], lw=3, label="Estimated2")
AXES.loglog(NVEC, np.mean(MISEAB, axis=0), 'r-', lw=3, label="Mean")
AXES.set_xlabel("Number of samples")
AXES.set_ylabel("MISE")
AXES.legend()
AXES.grid(True)

_, (AX1, AX2) = plt.subplots(2, 1, figsize=(7, 5))
for i, (AXES, mise) in enumerate(zip([AX1, AX2], [MISEA, MISEB])):
    AXES.loglog(NVEC, mise[0], lw=3, label="Estimated a")
    AXES.legend()
    LOGFIT = np.polyfit(np.log(NVEC), np.log(mise[0]), 1)
    AXES.set_title("Approximation of MISE{:d}: {:.3f}*n^{:.2f}".format(i, np.exp(LOGFIT[1]),
                                                                       LOGFIT[0]))
    AXES.set_xlabel("Number of samples")
    AXES.set_ylabel("MISE")
    AXES.grid(True)
plt.tight_layout()

LOGFIT = np.polyfit(np.log(NVEC), np.log(MISE[0]), 1)
print("Formula for estimated MISE assuming full dependecy: {:.3f}*n^{:.2f}".format(
    np.exp(LOGFIT[1]), LOGFIT[0]))
LOGFIT = np.polyfit(np.log(NVEC), np.log(MISEAB[0]), 1)
print("Formula for estimated MISE without this assumption: {:.3f}*n^{:.2f}".format(
    np.exp(LOGFIT[1]), LOGFIT[0]))
plt.show()
