import numpy as np
import matplotlib.pyplot as plt
import skimage as ski

def analyze_statistics(project):
    np.random.seed(42)
    # Extract the areas of the detected structures
    areas = np.array([prop.area for prop in project.properties])

    # Calculate the observed mean area
    observed_mean_area = np.mean(areas)

    print(f"Observed mean area: {observed_mean_area:.0f} pixels")

    # Bootstrap analysis of the mean area
    bootstrap_means = []

    ''' Create 1000 bootstrap samples from measured structure areas, structure can be selected more than once. 
    This allows us to estimate how much the calculated mean area would vary if we repeatedly sampled from the
    observed structures'''

    for i in range(1000):
        sample = np.random.choice(areas, size=len(areas), replace=True)

        bootstrap_means.append(np.mean(sample))

    bootstrap_sd = np.std(bootstrap_means)

    print(f"Bootstrap Standard Deviation of mean area: {bootstrap_sd:.2f} pixels")

    plt.figure(figsize=(8, 5))
    plt.hist(bootstrap_means, bins=30, edgecolor='black', alpha=0.8)
    plt.axvline(observed_mean_area, linestyle='--', linewidth=2)
    plt.title("Bootstrap Distribution of Mean Structure Area")
    plt.xlabel("mean area (pixels)")
    plt.ylabel("Frequency")
    plt.show()

    ''' If the original 1D signal is randomly rearranged, 
        do we still obtain a similar number or detected structures?'''

    shuffle_counts = []

    for i in range(100):
        shuffled_signal = np.random.permutation(project.raw_1d_signal) 
        ''' Keep all the original values but randomly rearrange their order.
            Therefore, the signal's individual amplitudes are preserved, but 
            its original temporal structure is destoryed '''
        # Fold each shuffled signal into the same 1024 x 1024 field    
        shuffled_image = shuffled_signal.reshape((project.resolution, project.resolution))
        shuffled_norm = (shuffled_image - np.min(shuffled_image)) / (np.max(shuffled_image) - np.min(shuffled_image))
        shuffled_processed = ski.exposure.equalize_adapthist(shuffled_norm)

        # Same Fourier low-pass reconstriction to each shuffled image
        shuffled_fft = np.fft.fftshift(np.fft.fft2(shuffled_processed))
        shuffled_fft = shuffled_fft * project.mask
        shuffled_reconstructed = np.abs(np.fft.ifft2(np.fft.ifftshift(shuffled_fft)))

        # Same segmentation preparation as the real data
        shuffled_smoothed = ski.filters.gaussian(shuffled_reconstructed, sigma=5)
        shuffled_thresh = ski.filters.threshold_otsu(shuffled_smoothed)
        shuffled_mask = shuffled_smoothed > shuffled_thresh

        # Finishing the structure counting for one randomized signal and store in shuffle_counts
        shuffled_labeled = ski.measure.label(shuffled_mask)
        shuffled_properties = ski.measure.regionprops(shuffled_labeled)
        shuffle_counts.append(len(shuffled_properties))

    # Convert the 100 counts into numpy array plus mean and spread of the randomized null distribution
    shuffle_counts = np.array(shuffle_counts)

    print(f"Randomized mean structure count: {np.mean(shuffle_counts):.2f}")
    print(f"Randomized Standard Deviation: {np.std(shuffle_counts):.2f}")

    # Comparision with the real result
    real_count = len(project.properties)
    # +1 correction prevents p-value of exactly zero with a finite number of randomization
    p_value = (np.sum(shuffle_counts <= real_count) + 1) / (len(shuffle_counts) + 1)

    print(f"Empirical p-value: {p_value:.4f}")

    # Null distribution histogram with result marking
    plt.figure(figsize=(8, 5))
    plt.hist(shuffle_counts, bins=np.arange(shuffle_counts.min() - 0.5, shuffle_counts.max() + 1.5, 1), edgecolor='black', alpha=0.8)
    plt.axvline(real_count, linestyle='--', linewidth=2)
    plt.title("Null Distribution of Randomized Structure Counts")
    plt.xlabel("Number of detected structures")
    plt.ylabel("Frequency")
    plt.show()

    # Phase randomization test
    phase_counts = []

    for i in range(100):
        # Fourier transform of the original signal
        fft_signal = np.fft.rfft(project.raw_1d_signal)

        # Preserve the original Fourier amplitudes
        amplitudes = np.abs(fft_signal)

        # Generate random phases
        random_phases = np.random.uniform(0, 2 * np.pi, len(fft_signal))

        # Keep the DC component unchanged
        random_phases[0] = 0

        # Construct the randomized Fourier signal
        randomized_fft = amplitudes * np.exp(1j * random_phases)

        # Transform back to the time domain
        randomized_signal = np.fft.irfft(randomized_fft, n=len(project.raw_1d_signal))

        # Fold phase-randomized 1D signal into the same field
        phase_image = randomized_signal.reshape((project.resolution, project.resolution))

        phase_norm = ((phase_image - np.min(phase_image)) / (np.max(phase_image) - np.min(phase_image)))

        phase_processed = ski.exposure.equalize_adapthist(phase_norm)

        # Apply the same Fourier low-pass reconstruction
        phase_fft = np.fft.fftshift(np.fft.fft2(phase_processed))

        phase_fft = phase_fft * project.mask

        phase_reconstructed = np.abs(np.fft.ifft2(np.fft.ifftshift(phase_fft)))

        # Count the connected structures
        phase_smoothed = ski.filters.gaussian(phase_reconstructed, sigma=5)

        phase_thresh = ski.filters.threshold_otsu(phase_smoothed)

        phase_mask = phase_smoothed > phase_thresh

        phase_labeled = ski.measure.label(phase_mask)

        phase_properties = ski.measure.regionprops(phase_labeled)

        phase_counts.append(len(phase_properties))

    phase_counts = np.array(phase_counts)

    print(f"Phase-randomized mean structure count: "f"{np.mean(phase_counts):.2f}")

    print(f"Phase-randomized Standard Deviation: "f"{np.std(phase_counts):.2f}")

    phase_p_value = ((np.sum(phase_counts <= real_count) + 1) / (len(phase_counts) + 1))

    print(f"Phase-randomized empirical p-value: "f"{phase_p_value:.4f}")

    plt.figure(figsize=(8, 5))

    plt.hist(phase_counts, bins=np.arange(phase_counts.min() - 0.5, phase_counts.max() + 1.5, 1), edgecolor='black', alpha=0.8)

    plt.axvline(real_count, linestyle='--', linewidth=2)

    plt.title("Null Distribution of Phase-Randomized Structure Counts")

    plt.xlabel("Number of detected structures")
    plt.ylabel("Frequency")

    plt.show()