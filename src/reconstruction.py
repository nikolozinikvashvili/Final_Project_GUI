import numpy as np
from scipy import fft 
import matplotlib.pyplot as plt

def reconstruct(project):
    f_transform = fft.fft2(project.processed_image)
    project.fft_shifted = fft.fftshift(f_transform)
    # 2D coordinate system centered at zero, circular mask is defined by distance from the centre x^2+y^2<=r^2
    # project.resolution is 1024 so the coordinates vary from -512 to 512
    y, x = np.ogrid[-project.resolution//2:project.resolution//2, -project.resolution//2:project.resolution//2]

    # Keeps low spatial frequencies near the centre of the Fourier spectrum and rejects high freq components
    project.mask = x*x + y*y <= 12**2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    # np.log1p by applying logarithmic scaling we can see weak and strong frequencies together
    ax1.imshow(np.log1p(np.abs(project.fft_shifted)), cmap='gray')
    ax1.set_title("2D Fourier Spectrum")
    ax1.axis('off')

    ax2.imshow(project.mask, cmap='gray')
    ax2.set_title("circular Low_Pass mask")
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

    # keeps low spatial frequencies inside the radius 12 circle and removes the outlayer
    filtered_fft = project.fft_shifted * project.mask

    # moves the filtered Fourier spectrum back to the arrangement required for the inverse Fourier transform
    filtered_fft = fft.ifftshift(filtered_fft)

    # to perform inverse 2D Fourier transform, bringing the filtered frequency domain data back into 2D domain
    project.reconstructed_cmb = np.abs(fft.ifft2(filtered_fft))

    # Final 2D field after low-pass filtering and inverse Fourier transform
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    ax1.imshow(project.reconstructed_cmb, cmap='gray')
    ax1.set_title("Reconstructed CMB Candidate Field", fontsize=14)
    ax1.axis('off')

    ax2.imshow(project.reconstructed_cmb, cmap='viridis')
    ax2.set_title("Reconstructed Candidate Field Viridis", fontsize=14)
    ax2.axis('off')

    plt.tight_layout()
    plt.show()








