"""
MISE bivariate

This script produces figures that are used for the completeness paper. Data from two different
distributions are created. These distributions are independent. Usnig the data,
the probability density functions (pdfs) are estimated: two univariate pdfs and one bivariate
pdf. The real mean integrated squared error (MISE) is computed of both these estimated and the
measure for completeness (see paper, this is an approximation of the real MISE) is computed.

Author
------
Erwin de Gelder

Creation
--------
09 Oct 2018


To do
-----


Modifications
-------------
01 Nov 2018 Create correct plots for the completeness paper.
07 Nov 2018 Make PEP8 compliant and update code based on update of fastkde.py.

"""

import os
import numpy as np
from matplotlib2tikz import save
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
from gaussianmixture import GaussianMixture
from fastkde import KDE

# Parameters
NMAX = 5000
NMIN = 100
NSTEPS = 50
NPDF = 101
SEED = 0
NREPEAT = 200
OVERWRITE = False
XLIM = [-3, 3]
MUA = [-1, 1]
MUB = [-0.5, 0.5, 1.5]
SIGMAA = [0.5, 0.3]
SIGMAB = [0.3, 0.5, 0.3]
MAX_CHANGE_BW = 0.25
FIGURES_FOLDER = os.path.join('..', '20180924 Completeness paper', 'figures')

# Create object for generating data from a Gaussian mixture
GMA = GaussianMixture(np.array(MUA), np.array(SIGMAA))
GMB = GaussianMixture(np.array(MUB), np.array(SIGMAB))

# Generate data
np.random.seed(SEED)
A = GMA.generate_samples(NMAX * NREPEAT).reshape((NREPEAT, NMAX))
B = GMB.generate_samples(NMAX * NREPEAT).reshape((NREPEAT, NMAX))
AB = np.concatenate((A[:, :, np.newaxis], B[:, :, np.newaxis]), axis=2)
NN = np.round(np.logspace(np.log10(NMIN), np.log10(NMAX), NSTEPS)).astype(np.int)

# Create the real pdfs
(XPDF,), YPDFA = GMA.pdf(minx=[XLIM[0]], maxx=[XLIM[1]], npoints=NPDF)
_, YPDFB = GMB.pdf(minx=[XLIM[0]], maxx=[XLIM[1]], npoints=NPDF)
YPDFAB = np.dot(YPDFB[:, np.newaxis], YPDFA[np.newaxis, :])
XPDFAB = np.transpose(np.array(np.meshgrid(XPDF, XPDF)), [1, 2, 0])

FILENAME = os.path.join("hdf5", "mise_bivariate.hdf5")
if OVERWRITE or not os.path.exists(FILENAME):
    BWA = np.zeros(len(NN))
    BWB = np.zeros_like(BWA)
    BWAB = np.zeros_like(BWA)

    # We use the first set of datapoints (i.e., X[0]) for determining the bandwidth
    # This has quite some influence on the remainder of the procedure...
    I_BW = 0

    # Compute the bandwidth
    print("Compute the bandwidths")
    KDEA = KDE(data=A[I_BW])
    KDEB = KDE(data=B[I_BW])
    KDEAB = KDE(data=AB[I_BW])
    for i, n in enumerate(tqdm(NN)):
        # Compute the optimal bandwidth using 1-leave-out
        KDEA.set_n(n)
        KDEB.set_n(n)
        KDEAB.set_n(n)
        if i == 0:
            KDEA.compute_bandwidth()
            KDEB.compute_bandwidth()
            KDEAB.compute_bandwidth()
        else:
            KDEA.compute_bandwidth(min_bw=KDEA.bandwidth * (1 - MAX_CHANGE_BW),
                                   max_bw=KDEA.bandwidth * (1 + MAX_CHANGE_BW))
            KDEB.compute_bandwidth(min_bw=KDEB.bandwidth * (1 - MAX_CHANGE_BW),
                                   max_bw=KDEB.bandwidth * (1 + MAX_CHANGE_BW))
            KDEAB.compute_bandwidth(min_bw=KDEAB.bandwidth * (1 - MAX_CHANGE_BW),
                                    max_bw=KDEAB.bandwidth * (1 + MAX_CHANGE_BW))
        BWA[i] = KDEA.bandwidth
        BWB[i] = KDEB.bandwidth
        BWAB[i] = KDEAB.bandwidth

    # Get the constants mu_k
    MUKA = KDEA.constants['muk']
    MUKAB = KDEAB.constants['muk']

    # Compute all the nrepeats pdfs. We need to do this many times in order to determine the real
    # MISE.
    # At the same time (to save memory), the MISE is computed
    # Real MISE for multivariate KDE, i.e., data assumed to be dependent
    REAL_MISE_DEP = np.zeros((len(NN), NREPEAT))
    EST_MISE_DEP = np.zeros((len(NN), NREPEAT))
    # Real MISE for when data is assumed to be independent
    REAL_MISE_IND = np.zeros_like(REAL_MISE_DEP)
    EST_MISE_IND = np.zeros_like(EST_MISE_DEP)
    print("Estimate pdfs")
    for j in tqdm(range(NREPEAT)):
        # Initialize the KDE
        KDEA = KDE(data=A[j])
        KDEB = KDE(data=B[j])
        KDEAB = KDE(data=AB[j])
        KDEA.set_score_samples(XPDF)
        KDEB.set_score_samples(XPDF)
        KDEAB.set_score_samples(XPDFAB)

        for i, n in enumerate(NN):
            # Compute the pdfs and the Laplacians
            KDEA.set_n(n)
            KDEA.set_bandwidth(BWA[i])
            PDFA_EST = KDEA.score_samples()
            LAPLACIANA = KDEA.laplacian()
            KDEB.set_n(n)
            KDEB.set_bandwidth(BWB[i])
            PDFB_EST = KDEB.score_samples()
            LAPLACIANB = KDEB.laplacian()
            KDEAB.set_n(n)
            KDEAB.set_bandwidth(BWAB[i])
            PDFAB_EST = KDEAB.score_samples()
            LAPLACIANAB = KDEAB.laplacian()

            # Compute the real MISE and estimate the MISE of the bivariate distribution
            REAL_MISE_DEP[i, j] = np.trapz(np.trapz((PDFAB_EST - YPDFAB) ** 2, XPDF), XPDF)
            INTEGRAL_LAPLACIAN = np.trapz(np.trapz(LAPLACIANAB ** 2, XPDF), XPDF)
            EST_MISE_DEP[i, j] = MUKAB / (n * BWAB[i] ** 2) + INTEGRAL_LAPLACIAN * BWAB[i] ** 4 / 4

            # Estimate the real MISE by estimating the probability by pa*pb (i.e., data assumed to
            # be independent)
            PDFAB_IND = np.dot(PDFB_EST[:, np.newaxis], PDFA_EST[np.newaxis, :])
            REAL_MISE_IND[i, j] = np.trapz(np.trapz((PDFAB_IND - YPDFAB) ** 2, XPDF), XPDF)

            # Compute the MISE for pdf a and b
            integrala = np.trapz(LAPLACIANA ** 2, XPDF)
            est_misea = MUKA / (n * BWA[i]) + integrala * BWA[i] ** 4 / 4
            integralb = np.trapz(LAPLACIANB ** 2, XPDF)
            est_miseb = MUKA / (n * BWB[i]) + integrala * BWB[i] ** 4 / 4

            # Compute the integrals of the squared probabilities
            integral_sqya = np.trapz(PDFA_EST ** 2, XPDF)
            integral_sqyb = np.trapz(PDFB_EST ** 2, XPDF)

            # Finally, estimate the MISE
            EST_MISE_IND[i, j] = (integral_sqyb * est_misea + integral_sqya * est_miseb +
                                  est_misea * est_miseb)

    # Write results to hdf5 file
    with h5py.File(FILENAME, "w") as f:
        f.create_dataset("real_mise_dep", data=REAL_MISE_DEP)
        f.create_dataset("est_mise_dep", data=EST_MISE_DEP)
        f.create_dataset("real_mise_ind", data=REAL_MISE_IND)
        f.create_dataset("est_mise_ind", data=EST_MISE_IND)
        f.create_dataset("bwa", data=BWA)
        f.create_dataset("bwb", data=BWB)
        f.create_dataset("bwab", data=BWAB)
else:
    with h5py.File(FILENAME, "r") as f:
        BWA = f["bwa"][:]
        BWB = f["bwb"][:]
        BWAB = f["bwab"][:]
        REAL_MISE_DEP = f["real_mise_dep"][:]
        EST_MISE_DEP = f["est_mise_dep"][:]
        REAL_MISE_IND = f["real_mise_ind"][:]
        EST_MISE_IND = f["est_mise_ind"][:]

# Compute power by which the MISE is decreasing
REAL_MISE_DEP = np.mean(REAL_MISE_DEP, axis=1)
REAL_MISE_IND = np.mean(REAL_MISE_IND, axis=1)
print("Power for real MISE dependent:        {:.2f}".
      format(np.polyfit(np.log(NN), np.log(REAL_MISE_DEP), 1)[0]))
print("Power for estimated MISE dependent:   {:.2f}".
      format(np.polyfit(np.log(NN), np.log(EST_MISE_DEP[:, 0]), 1)[0]))
print("Power for real MISE independent:      {:.2f}".
      format(np.polyfit(np.log(NN), np.log(REAL_MISE_IND), 1)[0]))
print("Power for estimated MISE independent: {:.2f}".
      format(np.polyfit(np.log(NN), np.log(EST_MISE_IND[:, 0]), 1)[0]))

# Plot the MISEs
SIGMAS = 3
_, AX = plt.subplots(1, 1, figsize=(7, 5))
PLOTREAL = AX.loglog(NN, REAL_MISE_DEP, lw=5, label="Real MISE", color=[0, 0, 0])
PLOTEST = AX.loglog(NN, EST_MISE_DEP[:, 0], lw=5, label="Estimated MISE", color=[.5, .5, .5])
MEAN = np.mean(EST_MISE_DEP, axis=1)
STD = np.std(EST_MISE_DEP, axis=1)
AX.fill_between(NN, MEAN - 3 * STD, MEAN + 3 * STD, color=[.8, .8, .8])
AX.loglog(NN, REAL_MISE_IND, '--', lw=5, color=PLOTREAL[0].get_color())
AX.loglog(NN, EST_MISE_IND[:, 0], '--', lw=5, color=PLOTEST[0].get_color())
MEAN = np.mean(EST_MISE_IND, axis=1)
STD = np.std(EST_MISE_IND, axis=1)
AX.fill_between(NN, MEAN - 3 * STD, MEAN + 3 * STD, color=[.8, .8, .8])
AX.set_xlabel("Number of samples")
# ax.set_ylabel("MISE")
# ax.grid(True)
AX.set_xlim([np.min(NN), np.max(NN)])
# ax.set_title("Data to be assumed dependent (solid) or independent (dashed)")
save(os.path.join(FIGURES_FOLDER, 'mise_example.tikz'),
     figureheight='\\figureheight', figurewidth='\\figurewidth')

# Plot the bandwidth for varying n
DX = np.mean(np.gradient(XPDF))
LAPA = np.trapz((np.gradient(np.gradient(YPDFA)) / DX ** 2) ** 2, XPDF)
MUKA = 1 / (2 * np.sqrt(np.pi))
HOPTA = (1 * MUKA / (NN * LAPA)) ** (1 / (1 + 4))
LAPB = np.trapz((np.gradient(np.gradient(YPDFB)) / DX ** 2) ** 2, XPDF)
HOPTB = (1 * MUKA / (NN * LAPB)) ** (1 / (1 + 4))
LAPAB = np.trapz(np.trapz((np.gradient(np.gradient(YPDFAB, axis=0), axis=0) / DX ** 2 +
                           np.gradient(np.gradient(YPDFAB, axis=1), axis=1) / DX ** 2) ** 2, XPDF),
                 XPDF)
print("Laplacian AB: {:.5f}".format(LAPAB))
MUKAB = 1 / (4 * np.pi)
HOPTAB = ((2 * MUKAB) / (NN * LAPAB)) ** (1 / (2 + 4))

_, AX = plt.subplots(1, 1, figsize=(7, 5))
AX.loglog(NN, BWA, '-', label="bwa", lw=5, color=[.5, .5, .5])
# ax.loglog(nn, hopta)
AX.loglog(NN, BWB, ':', label="bwb", lw=5, color=[.5, .5, .5])
# ax.loglog(nn, hoptb)
AX.loglog(NN, BWAB, '--', label="bwab", lw=5, color=[0, 0, 0])
# ax.loglog(nn, hoptab)
AX.set_xlabel("Number of samples")
AX.set_ylabel("Bandwidth")
# ax.grid(True)
AX.set_xlim([np.min(NN), np.max(NN)])
save(os.path.join(FIGURES_FOLDER, 'bandwidth.tikz'),
     figureheight='\\figureheight', figurewidth='\\figurewidth')

# Make a figure of the distribution and export to tikz
# To get the xticks right, use more datapoints
NPDF = 601
(XPDF,), YPDFA = GMA.pdf(minx=[XLIM[0]], maxx=[XLIM[1]], npoints=NPDF)
_, YPDFB = GMB.pdf(minx=[XLIM[0]], maxx=[XLIM[1]], npoints=NPDF)
_, AX = plt.subplots(1, 1, figsize=(7, 5))
AX.plot(XPDF, YPDFA, lw=5, color=[0, 0, 0])
AX.plot(XPDF, YPDFB, '--', lw=5, color=[.5, .5, .5])
# ax.grid(True)
AX.set_xlabel('$x$')
AX.set_xlim(XLIM)
save(os.path.join(FIGURES_FOLDER, 'true_pdf.tikz'),
     figureheight='\\figureheight', figurewidth='\\figurewidth')
plt.show()
