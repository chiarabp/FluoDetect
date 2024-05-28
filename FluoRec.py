import pandas as pd
import numpy as np


class FluoRec(object):
    def __init__(self, db_name, id_, name, pks_num, pks_wav, fwhm, min_peak_height, score=None):
        self.db_name = db_name
        self.id_ = id_
        self.name = name
        self.pks_num = pks_num
        self.pks_wav = pks_wav
        self.fwhm = fwhm
        self.min_peak_height = min_peak_height
        self.score = score

    def get_spectra_high_res(self):
        spectrum_high_res = pd.read_csv(f"high_res_spectra_db/{self.db_name}/" + str(self.id_) + '.txt', sep=";", header=0)
        spectrum_high_res.columns = ['wavelengths', "intensities"]
        wavelengths = spectrum_high_res['wavelengths'].array
        intensities = spectrum_high_res['intensities'].array

        return wavelengths, intensities

    def get_spectra_dim_reduction(self, alpha):
        spectrum_high_res = pd.read_csv(f"high_res_spectra_db/{self.db_name}/" + str(self.id_) + '.txt', sep=";",
                                        header=0)
        spectrum_high_res.columns = ['wavelengths', "intensities"]
        wavelengths = spectrum_high_res['wavelengths'].array
        intensities = spectrum_high_res['intensities'].array

        # dimensionality reduction
        indices_remove = []
        for i in range(0, len(wavelengths), alpha):
            indices_remove.append(i)

        np.delete(intensities, indices_remove)
        np.delete(wavelengths, indices_remove)

        return wavelengths, intensities
