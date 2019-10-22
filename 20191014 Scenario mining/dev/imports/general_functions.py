import numpy as np
import matplotlib.pyplot as plt
import imageio

###########################################################################################################################
# RENDERING FUNCTION
###########################################################################################################################

def render(img, title=None, saveFig=False, outPath="./", formatFile=".jpg") :
    if saveFig :
        # save it as uint8 => remultiply by 255!
        if np.max(img) - np.min(img) == 1.0 :
            img *= 255
        img = img.astype(np.uint8)
        imageio.imwrite(outPath + title + formatFile, img)
    else :
        if np.max(img) > 1 :
            img /= 255
    
        plt.imshow(img)
        if title :
            plt.title(title)
    
        plt.show()
        
def print_valid_relative_positions(vrp) :
    print("valid relative positions in the dictionary:")
    for k in vrp :
        print(k)
        for k2 in vrp[k] :
            print("\t", k2)