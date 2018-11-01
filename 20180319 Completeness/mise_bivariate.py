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
nsteps = 50
npdf = 101
seed = 0
nrepeat = 200
overwrite = False
xlim = [-3, 3]
mua = [-1, 1]
mub = [-0.5, 0.5, 1.5]
sigmaa = [0.5, 0.3]
sigmab = [0.3, 0.5, 0.3]
max_change_bw = 0.25
figures_folder = os.path.join('..', '20180924 Completeness paper', 'figures')

# Create object for generating data from a Gaussian mixture
gma = GaussianMixture(mua, sigmaa)
gmb = GaussianMixture(mub, sigmab)

# Generate data
np.random.seed(seed)
A = gma.generate_samples(nmax*nrepeat).reshape((nrepeat, nmax))
B = gmb.generate_samples(nmax*nrepeat).reshape((nrepeat, nmax))
AB = np.concatenate((A[:, :, np.newaxis], B[:, :, np.newaxis]), axis=2)
nn = np.round(np.logspace(np.log10(nmin), np.log10(nmax), nsteps)).astype(np.int)

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

    # Compute the bandwidth
    print("Compute the bandwidths")
    kdea = KDE(data=A[i_bw])
    kdeb = KDE(data=B[i_bw])
    kdeab = KDE(data=AB[i_bw])
    for i, n in enumerate(tqdm(nn)):
        # Compute the optimal bandwidth using 1-leave-out
        kdea.set_n(n)
        kdeb.set_n(n)
        kdeab.set_n(n)
        if i == 0:
            kdea.compute_bw()
            kdeb.compute_bw()
            kdeab.compute_bw()
        else:
            kdea.compute_bw(min_bw=kdea.bw*(1-max_change_bw), max_bw=kdea.bw*(1+max_change_bw))
            kdeb.compute_bw(min_bw=kdeb.bw*(1-max_change_bw), max_bw=kdeb.bw*(1+max_change_bw))
            kdeab.compute_bw(min_bw=kdeab.bw*(1-max_change_bw), max_bw=kdeab.bw*(1+max_change_bw))
        bwa[i] = kdea.bw
        bwb[i] = kdeb.bw
        bwab[i] = kdeab.bw

    # Get the constants mu_k
    muka = kdea.muk
    mukab = kdeab.muk

    # Compute all the nrepeats pdfs. We need to do this many times in order to determine the real MISE.
    # At the same time (to save memory), the MISE is computed
    real_mise_dep = np.zeros((len(nn), nrepeat))  # Real MISE for multivariate KDE, i.e., data assumed to be dependent
    est_mise_dep = np.zeros((len(nn), nrepeat))
    real_mise_ind = np.zeros_like(real_mise_dep)  # Real MISE for when data is assumed to be independent
    est_mise_ind = np.zeros_like(est_mise_dep)
    print("Estimate pdfs")
    for j in tqdm(range(nrepeat)):
        # Initialize the KDE
        kdea = KDE(data=A[j])
        kdeb = KDE(data=B[j])
        kdeab = KDE(data=AB[j])
        kdea.set_score_samples(xpdf)
        kdeb.set_score_samples(xpdf)
        kdeab.set_score_samples(xpdfab)

        for i, n in enumerate(nn):
            # Compute the pdfs and the Laplacians
            kdea.set_n(n)
            kdea.set_bw(bwa[i])
            pdfa_est = kdea.score_samples()
            laplaciana = kdea.laplacian()
            kdeb.set_n(n)
            kdeb.set_bw(bwb[i])
            pdfb_est = kdeb.score_samples()
            laplacianb = kdeb.laplacian()
            kdeab.set_n(n)
            kdeab.set_bw(bwab[i])
            pdfab_est = kdeab.score_samples()
            laplacianab = kdeab.laplacian()

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
sigmas = 3
f, ax = plt.subplots(1, 1, figsize=(7, 5))
plotreal = ax.loglog(nn, real_mise_dep, lw=5, label="Real MISE", color=[0, 0, 0])
plotest = ax.loglog(nn, est_mise_dep[:, 0], lw=5, label="Estimated MISE", color=[.5, .5, .5])
mean = np.mean(est_mise_dep, axis=1)
std = np.std(est_mise_dep, axis=1)
ax.fill_between(nn, mean - 3 * std, mean + 3 * std, color=[.8, .8, .8])
ax.loglog(nn, real_mise_ind, '--', lw=5, color=plotreal[0].get_color())
ax.loglog(nn, est_mise_ind[:, 0], '--', lw=5, color=plotest[0].get_color())
mean = np.mean(est_mise_ind, axis=1)
std = np.std(est_mise_ind, axis=1)
ax.fill_between(nn, mean - 3 * std, mean + 3 * std, color=[.8, .8, .8])
ax.set_xlabel("Number of samples")
# ax.set_ylabel("MISE")
# ax.grid(True)
ax.set_xlim([np.min(nn), np.max(nn)])
# ax.set_title("Data to be assumed dependent (solid) or independent (dashed)")
save(os.path.join(figures_folder, 'mise_example.tikz'),
     figureheight='\\figureheight', figurewidth='\\figurewidth')

# Plot the bandwidth for varying n
dx = np.mean(np.gradient(xpdf))
lapa = np.trapz((np.gradient(np.gradient(ypdfa)) / dx**2)**2, xpdf)
muka = 1 / (2*np.sqrt(np.pi))
hopta = (1*muka / (nn*lapa))**(1/(1+4))
lapb = np.trapz((np.gradient(np.gradient(ypdfb)) / dx**2)**2, xpdf)
hoptb = (1*muka / (nn*lapb))**(1/(1+4))
lapab = np.trapz(np.trapz((np.gradient(np.gradient(ypdfab, axis=0), axis=0) / dx**2 +
                           np.gradient(np.gradient(ypdfab, axis=1), axis=1) / dx**2)**2, xpdf), xpdf)
print("Laplacian AB: {:.5f}".format(lapab))
mukab = 1 / (4*np.pi)
hoptab = ((2*mukab) / (nn*lapab))**(1/(2+4))

_, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.loglog(nn, bwa, '-', label="bwa", lw=5, color=[.5, .5, .5])
# ax.loglog(nn, hopta)
ax.loglog(nn, bwb, ':', label="bwb", lw=5, color=[.5, .5, .5])
# ax.loglog(nn, hoptb)
ax.loglog(nn, bwab, '--', label="bwab", lw=5, color=[0, 0, 0])
# ax.loglog(nn, hoptab)
ax.set_xlabel("Number of samples")
ax.set_ylabel("Bandwidth")
# ax.grid(True)
ax.set_xlim([np.min(nn), np.max(nn)])
save(os.path.join(figures_folder, 'bandwidth.tikz'),
     figureheight='\\figureheight', figurewidth='\\figurewidth')

# Make a figure of the distribution and export to tikz
_, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(xpdf, ypdfa, lw=5, color=[0, 0, 0])
ax.plot(xpdf, ypdfb, '--', lw=5, color=[.5, .5, .5])
# ax.grid(True)
ax.set_xlabel('$x$')
ax.set_xlim(xlim)
save(os.path.join(figures_folder, 'true_pdf.tikz'),
     figureheight='\\figureheight', figurewidth='\\figurewidth')
plt.show()
