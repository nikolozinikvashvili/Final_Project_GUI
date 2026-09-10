class CMBReconstructor:
    def __init__(self, resolution=1024):
        self.resolution = resolution
        self.target_samples = resolution * resolution
        self.raw_1d_signal = None
        self.processed_image = None
        self.fft_shifted = None
        self.mask = None
        self.reconstructed_cmb = None
        self.properties = None
