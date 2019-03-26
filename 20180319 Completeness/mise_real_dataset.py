import numpy as np
import pickle
import matplotlib.pyplot as plt
from fastkde import KDE
import os
from tqdm import tqdm_notebook as tqdm
import h5py
import psutil
from time import time
from matplotlib2tikz import save

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
da = 1
db = d - da
overwrite = False
overwrite2 = False
max_change_bw = 0.5
figures_folder = os.path.join('..', '20180924 Completeness paper', 'figures')

# Initialize some stuff
nn = np.round(np.logspace(np.log10(nmin), np.log10(nmax), nsteps)).astype(np.int)
bw = np.zeros((nrepeat, len(nn)))
mise_var = np.zeros_like(bw)
mise_bias = np.zeros_like(bw)
mise = np.zeros_like(bw)
bwa = np.zeros((nrepeat, len(nn)))
misea = np.zeros_like(bw)
bwb = np.zeros((nrepeat, len(nn)))
miseb = np.zeros_like(bw)
miseab = np.zeros_like(bw)
x = np.linspace(xmin, xmax, nx)
xx = np.transpose(np.meshgrid(*[x for _ in range(d)]), np.concatenate((np.arange(1, d + 1), [0])))
xxa = np.transpose(np.meshgrid(*[x for _ in range(da)]), np.concatenate((np.arange(1, da + 1), [0])))
xxb = np.transpose(np.meshgrid(*[x for _ in range(db)]), np.concatenate((np.arange(1, db + 1), [0])))
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
            print("{:8.1f} Step {:d} for {:d}, n={:d}/{:d}".format(time() - starttime, i + 1, len(nn), n, nmax))
            # Compute the bandwidth only if this is the first time
            kde.set_n(n)
            print("{:8.1f} Compute bandwidth".format(time() - starttime), end="")
            if i == 0:
                kde.compute_bandwidth()
            else:
                kde.compute_bandwidth(min_bw=kde.bandwidth * (1 - max_change_bw), max_bw=kde.bandwidth * (1 + max_change_bw))
            bw[repeat, i] = kde.bandwidth
            print(" --> {:.6f}".format(kde.bandwidth))

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

            print("Estimated total time: {:.1f} s".format((time() - starttime) * np.sum(nn) / np.sum(nn[:i + 1])))

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

np.random.seed(seed)
data = scaling.copy()
filename = os.path.join('hdf5', 'mise_real_dataset_split.hdf5')
if overwrite or not os.path.exists(filename):
    for repeat in tqdm(range(nrepeat)):
        np.random.shuffle(data)
        kdea = KDE(data=data[:, :da])
        kdeb = KDE(data=data[:, da:])
        kdea.set_score_samples(xxa)
        kdeb.set_score_samples(xxb)

        # Evaluate MISE for different values of n
        starttime = time()
        for i, n in enumerate(nn):
            print("{:8.1f} Step {:d} for {:d}, n={:d}/{:d}".format(time() - starttime, i + 1, len(nn), n, nmax))
            # Compute the bandwidth only if this is the first time
            kdea.set_n(n)
            kdeb.set_n(n)
            print("{:8.1f} Compute bandwidth".format(time() - starttime), end="")
            if i == 0:
                kdea.compute_bandwidth()
                kdeb.compute_bandwidth()
            else:
                kdea.compute_bandwidth(min_bw=kdea.bandwidth * (1 - max_change_bw), max_bw=kdea.bandwidth * (1 + max_change_bw))
                kdeb.compute_bandwidth(min_bw=kdeb.bandwidth * (1 - max_change_bw), max_bw=kdeb.bandwidth * (1 + max_change_bw))
            bwa[repeat, i] = kdea.bandwidth
            bwb[repeat, i] = kdeb.bandwidth
            print(" --> {:.6f} and {:.6f}".format(kdea.bandwidth, kdeb.bandwidth))

            # Compute the pdf
            print("{:8.1f} Compute pdf".format(time() - starttime))
            pdfa = kdea.score_samples()
            pdfb = kdeb.score_samples()
            intpdfa = pdfa ** 2
            intpdfb = pdfb ** 2
            for _ in range(da):
                intpdfa = np.trapz(intpdfa, x)
            for _ in range(db):
                intpdfb = np.trapz(intpdfb, x)

            # Compute integral of Laplacian
            print("{:8.1f} Compute Laplacian".format(time() - starttime))
            laplaciana = kdea.laplacian()
            laplacianb = kdeb.laplacian()
            # print("{:8.1f} Compute integral Laplacian".format(time() - starttime))
            integrala = laplaciana ** 2
            integralb = laplacianb ** 2
            for _ in range(da):
                integrala = np.trapz(integrala, x)
            for _ in range(db):
                integralb = np.trapz(integralb, x)

            # Estimate MISE
            # print("{:8.1f} Estimate MISE".format(time() - starttime))
            misea[repeat, i] = kdea.muk / (n * bwa[repeat, i] ** da) + bwa[repeat, i] ** 4 / 4 * integrala
            miseb[repeat, i] = kdeb.muk / (n * bwb[repeat, i] ** db) + bwb[repeat, i] ** 4 / 4 * integralb
            miseab[repeat, i] = (intpdfb * misea[repeat, i] + intpdfa * miseb[repeat, i] +
                                 misea[repeat, i] * miseb[repeat, i])
            print("Mise = {:.6e}".format(miseab[repeat, i]))
            print("{:02d} Estimated total time: {:.1f} s".format(repeat+1,
                                                                 (time() - starttime) * np.sum(nn) /
                                                                 np.sum(nn[:i + 1])))

    # Write data to file
    with h5py.File(filename, "w") as f:
        f.create_dataset("miseab", data=miseab)
        f.create_dataset("misea", data=misea)
        f.create_dataset("miseb", data=miseb)
        f.create_dataset("bwa", data=bwa)
        f.create_dataset("bwb", data=bwb)
else:
    with h5py.File(filename, "r") as f:
        miseab = f["miseab"][:]
        misea = f["misea"][:]
        miseb = f["miseb"][:]
        bwa = f["bwa"][:]
        bwb = f["bwb"][:]

with open(os.path.join('pickles', 'df.p'), 'rb') as f:
    dfs, scaling = pickle.load(f)
scaling = scaling.T  # [time vstart vend]
scaling = scaling[scaling[:, 2] > 0, :]  # Remove full stops
scaling[:, 1] = scaling[:, 1] - scaling[:, 2]  # Now it becomes: [time deltav vend]
scaling[:, 0] = scaling[:, 1] / scaling[:, 0]  # Now it becomes: [deceleration deltav vend]
f, axs = plt.subplots(3, 1, figsize=(7, 5))
for i, ax in enumerate(axs):
    ax.hist(scaling[:, i], bins=20, color=[.5, .5, .5], edgecolor=[0, 0, 0])
    xlim = ax.get_xlim()
    ax.set_xlim([0, xlim[1]])
axs[0].set_xlabel('Average deceleration [m/s$^2$]')
axs[1].set_xlabel('Speed difference [m/s]')
axs[2].set_xlabel('End speed [m/s]')
for ax in axs:
    ax.set_ylabel('Frequency')
plt.tight_layout()
save(os.path.join(figures_folder, 'histogram.tikz'),
     figureheight='\\figureheight', figurewidth='\\figurewidth',
     extra_axis_parameters=['axis x line*=bottom', 'axis y line*=left'])

LW = 1.75
_, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.loglog(nn, bwa[0], '-', label="bwa", lw=LW, color=[.5, .5, .5])
ax.loglog(nn, bwb[0], ':', label="bwb", lw=LW, color=[.5, .5, .5])
ax.loglog(nn, bw[0], '--', label="bwab", lw=LW, color=[0, 0, 0])
ax.set_xlabel("Number of samples")
ax.set_ylabel("Bandwidth")
ax.set_xlim([np.min(nn), np.max(nn)])
save(os.path.join(figures_folder, 'bandwidth_real.tikz'),
     figureheight='\\figureheight', figurewidth='\\figurewidth',
     extra_axis_parameters=['xtick={600, 800, 1000, 1500, 2000, 2500}',
                            'xticklabels={600, 800, 1000, 1500, 2000, 2500}',
                            'axis x line*=bottom', 'axis y line*=left',
                            'every x tick/.style={black}', 'every y tick/.style={black}'])

_, ax = plt.subplots(1, 1, figsize=(7, 5))
logfit1 = np.polyfit(np.log(nn), np.log(mise[0]), 1)
ax.loglog(nn, np.exp(logfit1[1]) * nn ** logfit1[0], color=[0, 0, 0], lw=LW)
ax.loglog(nn, mise[0], lw=LW, color=[.5, .5, .5])
print("Approximation of MISE1: {:.3f}*n^{:.2f}".format(np.exp(logfit1[1]), logfit1[0]))
logfit2 = np.polyfit(np.log(nn), np.log(miseab[0]), 1)
ax.loglog(nn, np.exp(logfit2[1]) * nn ** logfit2[0], '--', color=[0, 0, 0], lw=LW)
ax.loglog(nn, miseab[0], '--', lw=LW, color=[.5, .5, .5])
print("Approximation of MISE1: {:.3f}*n^{:.2f}".format(np.exp(logfit2[1]), logfit2[0]))
n2 = 1000
n1 = (np.exp(logfit2[1] - logfit1[1])*n2**logfit2[0])**(1/logfit1[0])
print("Number of points for MISE 1 to be equal to MISE 2 at n={:d}: {:.0f}".format(n2, n1))
ax.set_xlim([np.min(nn), np.max(nn)])
ax.set_xlabel("Number of samples")
ax.set_ylabel("Measure for completeness")
save(os.path.join(figures_folder, 'mise_real.tikz'),
     figureheight='\\figureheight', figurewidth='\\figurewidth',
     extra_axis_parameters=['xtick={600, 800, 1000, 1500, 2000, 2500}',
                            'xticklabels={600, 800, 1000, 1500, 2000, 2500}',
                            'axis x line*=bottom', 'axis y line*=left',
                            'every x tick/.style={black}', 'every y tick/.style={black}'])

_, ax = plt.subplots(1, 1, figsize=(7, 5))
for m in mise:
    ax.loglog(nn, m, color=[.5, .5, 1])
ax.loglog(nn, mise[0], lw=3, label="Estimated")
ax.loglog(nn, np.mean(mise, axis=0), 'r-', lw=3, label="Mean")
ax.set_xlabel("Number of samples")
ax.set_ylabel("MISE")
ax.legend()
ax.grid(True)

_, ax = plt.subplots(1, 1, figsize=(7, 5))
for m in miseab:
    plt.loglog(nn, m, color=[.5, .5, 1])
ax.loglog(nn, miseab[0], lw=3, label="Estimated2")
ax.loglog(nn, np.mean(miseab, axis=0), 'r-', lw=3, label="Mean")
ax.set_xlabel("Number of samples")
ax.set_ylabel("MISE")
ax.legend()
ax.grid(True)

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5))
for i, (ax, m) in enumerate(zip([ax1, ax2], [misea, miseb])):
    ax.loglog(nn, m[0], lw=3, label="Estimated a")
    ax.legend()
    logfit = np.polyfit(np.log(nn), np.log(m[0]), 1)
    ax.set_title("Approximation of MISE{:d}: {:.3f}*n^{:.2f}".format(i, np.exp(logfit[1]), logfit[0]))
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("MISE")
    ax.grid(True)
f.tight_layout()

logfit = np.polyfit(np.log(nn), np.log(mise[0]), 1)
print("Formula for estimated MISE assuming full dependecy: {:.3f}*n^{:.2f}".format(np.exp(logfit[1]), logfit[0]))
logfit = np.polyfit(np.log(nn), np.log(miseab[0]), 1)
print("Formula for estimated MISE without this assumption: {:.3f}*n^{:.2f}".format(np.exp(logfit[1]), logfit[0]))
plt.show()
