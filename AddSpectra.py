import pandas as pd

from DB import DB
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy.signal import peak_widths
from scipy.interpolate import interp1d
from PeaksSpot import peaks_spot
import os


class AddSpectra(object):
    # Defaults
    # led_range = [400, 401] #led spectrum to remove
    # smoothing_wind = 15 #smoothing window
    # min_peak_height = 0.1 #minimum height for recognizing a peak as such
    # alpha = 3 #dimesionality reduction parameter (1/3 records are discarded)

    def __init__(self, led_range, smoothing_wind, min_peak_height, alpha, spectrum_to_read):
        self.spectrum_to_read = spectrum_to_read
        self.led_range = led_range
        self.smoothing_wind = smoothing_wind
        self.min_peak_height = min_peak_height
        self.alpha = alpha

    def analyze_spectra(self):
        spectrum_data = self.spectrum_to_read
        spectrum_data.columns = ['wavelengths', "intensities", "not interested", "not interested"]
        wavelengths = np.array(spectrum_data['wavelengths'].to_numpy(), copy=True)
        intensities = np.array(spectrum_data['intensities'].to_numpy(), copy=True)

        # 1. Remove LED source
        wavs_remove = self.led_range

        indices_remove = np.where((wavelengths >= wavs_remove[0]) & (wavelengths <= wavs_remove[-1]))
        intensities[indices_remove] = 0

        # 2. Smoothing
        intensities = savgol_filter(intensities, self.smoothing_wind, 3)

        # 3. Normalization
        intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))

        # 4. Dimesionality reduction
        indices_remove = []
        for i in range(0, len(wavelengths), self.alpha):
            indices_remove.append(i)
        intensities_reduced = intensities # tmp copy
        wavelengths_reduced = wavelengths # tmp copy

        np.delete(intensities_reduced, indices_remove)
        np.delete(wavelengths_reduced, indices_remove)

        # 5. Collection of data and access results
        locations, width_heights, left_ips, right_ips, fwhm = peaks_spot(wavelengths_reduced, intensities_reduced, self.min_peak_height)

        return wavelengths_reduced, intensities_reduced, wavelengths, intensities, locations, fwhm, width_heights, left_ips, right_ips

    # Saves spectra
    # locations and fwhm refer to the location of the peaks and their fwhm
    def save_spectra(self, db_name, spectra_name, wavelengths, intensities, locations, fwhm, min_peak_height):
        db_data_set = DB(db_name)
        id_ = db_data_set.add_record(spectra_name, locations, fwhm, min_peak_height)
        db_data_set.close()
        # Create DataFrame
        df_highres = pd.DataFrame({'Wavelength': wavelengths, 'Intensity': intensities})

        # Create folders if they don't exist
        highres_folder = f"high_res_spectra_db/{db_name}/"
        os.makedirs(highres_folder, exist_ok=True)

        # Save high resolution data
        filename_highres = os.path.join(highres_folder, f"{id_}.txt")
        df_highres.to_csv(filename_highres, sep=';', index=False)

        return id_