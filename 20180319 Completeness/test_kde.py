import matplotlib.pyplot as plt
import numpy as np
from stats import KDE, GaussianMixture

# Create a distribution that is a Gaussian mixture.
xlim = [-3, 3]
gm = GaussianMixture([-1, 1], [0.5, 0.3])
(xpdf,), ypdf = gm.pdf(minx=[xlim[0]], maxx=[xlim[1]], npoints=100)

# Create the KDE.
np.random.seed(0)  # Set seed for repeatability.
x = gm.generate_samples(100)  # Use 100 datapoints. Increase to have a more accurate KDE.
kde = KDE(data=x)
kde.set_bandwidth(kde.silverman())  # Use the Silverman's rule of thumb.
ypdf_estimated = kde.score_samples(xpdf)

# Plot the result.
plt.plot(xpdf, ypdf, label="Real")
plt.plot(xpdf, ypdf_estimated, label="KDE")
plt.legend()
plt.show()
