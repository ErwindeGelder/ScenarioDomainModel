import numpy as np
from gaussianmixture import GaussianMixture
import os
# from matplotlib2tikz import save
import h5py
from tqdm import tqdm
from fastkde import KDE
import matplotlib.pyplot as plt

# Parameters
nmax = 5000
nmin = 100
nstep = 100
npdf = 61
seed = 0
nrepeat = 100
overwrite = False
xlim = [-3, 3]
mua = [-1, 1]
mub = [-0.5, 0.5, 1.5]
sigmaa = [0.5, 0.3]
sigmab = [0.3, 0.5, 0.3]

# Create object for generating data from a Gaussian mixture
gma = GaussianMixture(mua, sigmaa)
gmb = GaussianMixture(mub, sigmab)

# Generate data
np.random.seed(seed)
A = gma.generate_samples(nmax*nrepeat).reshape((nrepeat, nmax))
B = gmb.generate_samples(nmax*nrepeat).reshape((nrepeat, nmax))
AB = np.concatenate((A[:, :, np.newaxis], B[:, :, np.newaxis]), axis=2)
nn = np.arange(nmin, nmax+nstep, nstep)

# Create the real pdfs
(xpdf,), ypdfa = gma.pdf(minx=[xlim[0]], maxx=[xlim[1]], n=npdf)
_, ypdfb = gmb.pdf(minx=[xlim[0]], maxx=[xlim[1]], n=npdf)
ypdfab = np.dot(ypdfb[:, np.newaxis], ypdfa[np.newaxis, :])
xpdfab = np.transpose(np.array(np.meshgrid(xpdf, xpdf)), [1, 2, 0])

filename = os.path.join("hdf5", "mise_bivariate.hdf5")
if overwrite or not os.path.exists(filename):
    bwa = np.zeros(len(nn))
    bwb = np.zeros_like(bwa)
    bwab = np.zeros_like(bwa)

    # We use the first set of datapoints (i.e., X[0]) for determining the bandwidth
    # This has quite some influence on the remainder of the procedure...
    i_bw = 0

    # For now, just set the data for all the kdes
    print("Initialize all {:d} KDEs".format(nrepeat))
    kdesa = [KDE(data=x) for x in A]
    kdesb = [KDE(data=x) for x in B]
    kdesab = [KDE(data=x) for x in AB]
    for kdea, kdeb, kdeab in tqdm(zip(kdesa, kdesb, kdesab)):
        kdea.set_score_samples(xpdf)
        kdeb.set_score_samples(xpdf)
        kdeab.set_score_samples(xpdfab)

    # Compute the bandwidths
    print("Compute the bandwidths")
    for i, n in enumerate(tqdm(nn)):
        # Compute the optimal bandwidth using 1-leave-out
        kdesa[i_bw].set_n(n)
        kdesa[i_bw].compute_bw()
        bwa[i] = kdesa[i_bw].bw
        kdesb[i_bw].set_n(n)
        kdesb[i_bw].compute_bw()
        bwb[i] = kdesb[i_bw].bw
        kdesab[i_bw].set_n(n)
        kdesab[i_bw].compute_bw()
        bwab[i] = kdesab[i_bw].bw

    # Get the constants mu_k
    muka = kdesa[0].muk
    mukab = kdesab[0].muk

    # Compute all the nrepeats pdfs. We need to do this many times in order to determine the real MISE.
    # At the same time (to save memory), the MISE is computed
    real_mise_dep = np.zeros((len(nn), nrepeat))  # Real MISE for multivariate KDE, i.e., data assumed to be dependent
    est_mise_dep = np.zeros((len(nn), nrepeat))
    real_mise_ind = np.zeros_like(real_mise_dep)  # Real MISE for when data is assumed to be independent
    est_mise_ind = np.zeros_like(est_mise_dep)
    print("Estimate pdfs")
    for j in tqdm(range(nrepeat)):
        for i, n in enumerate(nn):
            # Compute the pdfs and the Laplacians
            kdesa[j].set_n(n)
            kdesa[j].set_bw(bwa[i])
            pdfa_est = kdesa[j].score_samples()
            laplaciana = kdesa[j].laplacian()
            kdesb[j].set_n(n)
            kdesb[j].set_bw(bwb[i])
            pdfb_est = kdesb[j].score_samples()
            laplacianb = kdesb[j].laplacian()
            kdesab[j].set_n(n)
            kdesab[j].set_bw(bwab[i])
            pdfab_est = kdesab[j].score_samples()
            laplacianab = kdesab[j].laplacian()

            # Compute the real MISE and estimate the MISE of the bivariate distribution
            real_mise_dep[i, j] = np.trapz(np.trapz((pdfab_est - ypdfab) ** 2, xpdf), xpdf)
            integral_laplacian = np.trapz(np.trapz(laplacianab ** 2, xpdf), xpdf)
            est_mise_dep[i, j] = mukab / (n * bwab[i] ** 2) + integral_laplacian * bwab[i] ** 4 / 4

            # Estimate the real MISE by estimating the probability by pa*pb (i.e., data assumed to be independet)
            pdfab_ind = np.dot(pdfb_est[:, np.newaxis], pdfa_est[np.newaxis, :])
            real_mise_ind[i, j] = np.trapz(np.trapz((pdfab_ind - ypdfab) ** 2, xpdf), xpdf)

            # Compute the MISE for pdf a and b
            integrala = np.trapz(laplaciana ** 2, xpdf)
            est_misea = muka / (n * bwa[i]) + integrala * bwa[i] ** 4 / 4
            integralb = np.trapz(laplacianb ** 2, xpdf)
            est_miseb = muka / (n * bwb[i]) + integrala * bwb[i] ** 4 / 4

            # Compute the integrals of the squared probabilities
            integral_sqya = np.trapz(pdfa_est ** 2, xpdf)
            integral_sqyb = np.trapz(pdfb_est ** 2, xpdf)

            # Finally, estimate the MISE
            est_mise_ind[i, j] = integral_sqyb * est_misea + integral_sqya * est_miseb + est_misea * est_miseb

    # Write results to hdf5 file
    with h5py.File(filename, "w") as f:
        f.create_dataset("real_mise_dep", data=real_mise_dep)
        f.create_dataset("est_mise_dep", data=est_mise_dep)
        f.create_dataset("real_mise_ind", data=real_mise_ind)
        f.create_dataset("est_mise_ind", data=est_mise_ind)
        f.create_dataset("bwa", data=bwa)
        f.create_dataset("bwb", data=bwb)
        f.create_dataset("bwab", data=bwab)
else:
    with h5py.File(filename, "r") as f:
        bwa = f["bwa"][:]
        bwb = f["bwb"][:]
        bwab = f["bwab"][:]
        real_mise_dep = f["real_mise_dep"][:]
        est_mise_dep = f["est_mise_dep"][:]
        real_mise_ind = f["real_mise_ind"][:]
        est_mise_ind = f["est_mise_ind"][:]

# Compute power by which the MISE is decreasing
real_mise_dep = np.mean(real_mise_dep, axis=1)
real_mise_ind = np.mean(real_mise_ind, axis=1)
print("Power for real MISE dependent:        {:.2f}".format(np.polyfit(np.log(nn), np.log(real_mise_dep), 1)[0]))
print("Power for estimated MISE dependent:   {:.2f}".format(np.polyfit(np.log(nn), np.log(est_mise_dep[:, 0]), 1)[0]))
print("Power for real MISE independent:      {:.2f}".format(np.polyfit(np.log(nn), np.log(real_mise_ind), 1)[0]))
print("Power for estimated MISE independent: {:.2f}".format(np.polyfit(np.log(nn), np.log(est_mise_ind[:, 0]), 1)[0]))

# Plot the MISEs
f, ax = plt.subplots(1, 1, figsize=(7, 5))
plotreal = ax.loglog(nn, real_mise_dep, lw=5, label="Real MISE")
plotest = ax.loglog(nn, est_mise_dep[:, 0], lw=5, label="Estimated MISE")
ax.loglog(nn, real_mise_ind, '--', lw=5, color=plotreal[0].get_color())
ax.loglog(nn, est_mise_ind[:, 0], '--', lw=5, color=plotest[0].get_color())
ax.set_xlabel("Number of samples")
ax.set_ylabel("MISE")
ax.legend()
ax.grid(True)
ax.set_xlim([np.min(nn), np.max(nn)])
ax.set_title("Data to be assumed dependent (solid) or independent (dashed)")
plt.show()

# Plot the bandwidth for varying n
_, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.semilogx(nn, bwa, label="bwa", lw=5)
ax.semilogx(nn, bwb, label="bwb", lw=5)
ax.semilogx(nn, bwab, label="bwab", lw=5)
ax.set_xlabel("Number of samples")
ax.set_ylabel("Bandwidth")
ax.grid(True)
ax.legend()
ax.set_xlim([np.min(nn), np.max(nn)])
plt.show()
