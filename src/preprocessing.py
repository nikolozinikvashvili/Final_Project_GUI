import numpy as np 
import matplotlib.pyplot as plt
import skimage as ski
def preprocess(project):
    
    # Rearrange 1048576 samples into 1024x1024 matrix
    folded = project.raw_1d_signal.reshape((project.resolution, project.resolution)) 

    # Rescales the values of folded image respectively min and max to 0 and 1
    norm = (folded - np.min(folded)) / (np.max(folded) - np.min(folded)) 

    # Enhace local contrast of normalised 2D field.
    project.processed_image = ski.exposure.equalize_adapthist(norm)

    # Visualize the rusult
    plt.figure(figsize=(8, 8))

    # Because we have numerical intensity field
    plt.imshow(project.processed_image, cmap='gray')

    plt.title("2D folded Signal - Adaptive Histogram Equalization", fontsize=14)
    plt.axis('off') # axis doesn't represent a physical spatial coordinate but pixel position of field
    plt.show()
