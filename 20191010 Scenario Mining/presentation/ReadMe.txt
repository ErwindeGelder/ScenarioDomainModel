The following files need to be in the folder of your TeX file:

beamercolorthemeTNO.sty
beamerinnerthemeTNO.sty
beamerouterthemeTNO.sty
beamerthemeTNO.sty
TNObannerBlack.pdf
TNObannerWhite.pdf
TNOtopBannerBlue.pdf
TNOtopBannerWhite.pdf

Alternatively, you can make a folder in <TeXroot>\tex\latex, where TeXroot is typically C:\Program Files\MiKTeX 2.9\tex\latex, and copy the aforementioned files to this folder. Then start MiKTeX Settings (Admin) from the Windows Start menu, and press "Refresh FNDB" on the General tab.

The TeX file print2x2.tex is meant to print the presentation in a 2 x 2 format per page. Open this file, fill-in the requested information, and compile it using PDFTexify.