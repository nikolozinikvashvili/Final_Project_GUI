import numpy as np
import matplotlib.pyplot as plt
import skimage as ski

def segment(project):
    # Smooth the reconstructed field before detecting large-scale structures
    smoothed = ski.filters.gaussian(project.reconstructed_cmb, sigma=5)

    # Automatically determine the intensity threshold for structure detection
    thresh = ski.filters.threshold_otsu(smoothed)
    
    mask = smoothed > thresh

    labeled_image = ski.measure.label(mask)
    project.properties = ski.measure.regionprops(labeled_image)

    # vizualization
    plt.figure(figsize=(8, 8))
    plt.imshow(labeled_image, cmap='nipy_spectral')
    plt.title("Detected Structures")
    plt.axis('off')
    plt.show

    print(f"Number of detected structures: {len(project.properties)}")