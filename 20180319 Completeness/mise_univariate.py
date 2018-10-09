import numpy as np
from gaussianmixture import GaussianMixture
import os
from matplotlib2tikz import save
import h5py
from tqdm import tqdm
from fastkde import KDE
import matplotlib.pyplot as plt

# Parameters
nmax = 5000
nmin = 100
nstep = 100
seed = 1
overwrite = False
nrepeat = 200
npdf = 201
mu = [-1, 1]
sigma = [0.5, 0.3]
xlim = [-3, 3]
filename = os.path.join("hdf5", "mise_univariate.hdf5")

# Generate datapoints
np.random.seed(seed)
gm = GaussianMixture(mu, sigma)
X = gm.generate_samples(nmax*nrepeat).reshape((nrepeat, nmax))

# This we need for the next step
(xpdf,), ypdf = gm.pdf(minx=[xlim[0]], maxx=[xlim[1]], n=npdf)
nn = np.arange(nmin, nmax+nstep, nstep)
dx = np.mean(np.gradient(xpdf))
muk = 1 / (2 * np.sqrt(np.pi))

if overwrite or not os.path.exists(filename):
    real_mise = np.zeros(len(nn))
    est_mise = np.zeros((len(nn), nrepeat))

    kde_estimated = np.zeros((len(nn), nrepeat, len(xpdf)))
    laplacians = np.zeros_like(kde_estimated)
    bw = np.zeros(len(nn))

    # We use the first set of datapoints (i.e., X[0]) for determining the bandwidth
    # This has quite some influence on the remainder of the procedure...
    i_bw = 0

    # For now, just set the data for all the kdes
    print("Initialize all KDEs")
    kdes = [KDE(data=x) for x in X]
    for kde in kdes:
        kde.set_score_samples(xpdf)

    print("Compute the bandwidth for varying number of datapoints")
    for i, n in enumerate(tqdm(nn)):
        # Compute the optimal bandwidth using 1-leave-out
        kdes[i_bw].set_n(n)
        kdes[i_bw].compute_bw()
        bw[i] = kdes[i_bw].bw

    print("Estimate the pdfs and Laplacians")
    for j in tqdm(range(nrepeat)):
        for i, n in enumerate(nn):
            # Compute all the nrepeats pdfs. We need to do this many times in order to determine the real MISE
            kdes[j].set_n(n)
            kdes[j].set_bw(bw[i])
            kde_estimated[i, j] = kdes[j].score_samples()
            laplacians[i, j] = kdes[j].laplacian()

    print("Estimate the MISE")
    for i, (y, laplacian, h, n) in enumerate(zip(kde_estimated, laplacians, bw, nn)):
        real_mise[i] = np.trapz(np.mean((y - ypdf) ** 2, axis=0), xpdf)
        integral_laplacian = np.trapz(laplacian ** 2, xpdf)
        est_mise[i] = muk / (n * h) + integral_laplacian * h ** 4 / 4

    # Write results to hdf5 file
    with h5py.File(filename, "w") as f:
        f.create_dataset("real_mise", data=real_mise)
        f.create_dataset("est_mise", data=est_mise)
        f.create_dataset("bw", data=bw)
else:
    with h5py.File(filename, "r") as f:
        real_mise = f["real_mise"][:]
        est_mise = f["est_mise"][:]
        bw = f["bw"][:]

# Compute power by which the MISE is decreasing
print("Power for real MISE:      {:.2f}".format(np.polyfit(np.log(nn), np.log(real_mise), 1)[0]))
print("Power for estimated MISE: {:.2f}".format(np.polyfit(np.log(nn), np.log(est_mise[:, 0]), 1)[0]))

# Plot the real MISE and the estimated MISE and export plot to tikz
_, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.loglog(nn, real_mise, label="Real MISE", lw=5)
ax.loglog(nn, est_mise[:, 0], label="Estimated MISE", lw=5)
ax.set_xlabel("Number of samples")
ax.set_ylabel("MISE")
ax.legend()
ax.grid(True)
ax.set_xlim([np.min(nn), np.max(nn)])
save(os.path.join('..', '20181002 Completeness question', 'figures', 'mise_example.tikz'),
     figureheight='\\figureheight', figurewidth='\\figurewidth')
plt.show()

# Plot again the real MISE and the estimated MISE and export plot to tikz
_, ax = plt.subplots(1, 1, figsize=(7, 5))
est_mean = np.mean(est_mise, axis=1)
est_std = np.std(est_mise, axis=1)
ax.loglog(nn, real_mise, label="Real MISE", lw=5)
p = ax.loglog(nn, est_mean, label="Estimated MISE", lw=5)
ax.loglog(nn, est_mean+2*est_std, '--', color=p[0].get_color(), lw=5)
ax.loglog(nn, est_mean-2*est_std, '--', color=p[0].get_color(), lw=5)
ax.set_xlabel("Number of samples")
ax.set_ylabel("MISE")
ax.legend()
ax.grid(True)
ax.set_xlim([np.min(nn), np.max(nn)])
save(os.path.join('..', '20181002 Completeness question', 'figures', 'mise_example2.tikz'),
     figureheight='\\figureheight', figurewidth='\\figurewidth')
plt.show()

# Plot the bandwidth for varying n
_, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.semilogx(nn, bw)
ax.set_xlabel("Number of samples")
ax.set_ylabel("Bandwidth")
ax.grid(True)
ax.set_xlim([np.min(nn), np.max(nn)])
plt.show()
