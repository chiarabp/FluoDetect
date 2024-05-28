import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, peak_widths


def peaks_spot(wavelengths, intensities, min_peak_height):
    # 5. Collection of data
    peaks, _ = find_peaks(intensities, prominence=min_peak_height)
    results_half = peak_widths(intensities, peaks, rel_height=0.5)

    # Access results
    locations = wavelengths[peaks]
    width_heights = results_half[1]
    left_ips = index_to_xdata(wavelengths, results_half[2])
    right_ips = index_to_xdata(wavelengths, results_half[3])
    fwhm = right_ips - left_ips

    return locations, width_heights, left_ips, right_ips, fwhm


def index_to_xdata(xdata, indices):
    # interpolate the values from signal.peak_widths to xdata
    ind = np.arange(len(xdata))
    f = interp1d(ind,xdata)
    return f(indices)
