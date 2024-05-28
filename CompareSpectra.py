import numpy as np
import pandas as pd
from DB import DB
from FluoRec import FluoRec
from AddSpectra import AddSpectra


class CompareSpectra:
    # comp_set = {
    #    'pk_num_tolerance' = 4,
    #    'pk_tolerance' = 30,
    #    'fwhm_tolerance' = 50,
    #    'weight_peak' = 0.5,
    #    'weight_fwhm' = 0.2,
    #    'weight_shape' = 0.3,
    #    'num_records_to_show' = 5,
    #    'shape_pt_tol' = 0.3,
    #    'baseline_pts' = 0.1
    # }

    def __init__(self, db_name, led_range, smoothing_wind, min_peak_height, alpha, spectrum_to_read, comp_set):
        self.db_name = db_name
        self.spectrum_to_read = spectrum_to_read
        self.led_range = led_range
        self.smoothing_wind = smoothing_wind
        self.min_peak_height = min_peak_height
        self.alpha = alpha
        self.comp_set = comp_set

    def compare_fluorescence_recs(self, spectra_compare, fluo_rec):
        # Check number of peaks
        diff_num_pks = abs(spectra_compare.pks_num - fluo_rec.pks_num)
        score_num_peaks = 1 - diff_num_pks / self.comp_set['pk_num_tolerance']
        score_num_peaks = max(0, score_num_peaks)

        # Check wavelength of peaks
        matching_peak_index = np.full(spectra_compare.pks_num, -1)
        diff_peak = np.full(spectra_compare.pks_num, self.comp_set['pk_tolerance'])

        # identify matching peaks
        for i in range(spectra_compare.pks_num):
            for j in range(fluo_rec.pks_num):
                diff_peak_val = abs(spectra_compare.pks_wav[i] - fluo_rec.pks_wav[j])
                if diff_peak_val <= self.comp_set['pk_tolerance']:
                    diff_peak[i] = diff_peak_val
                    matching_peak_index[i] = j

        score_peaks_wavs = np.full(len(diff_peak), 0, np.float_)

        # assign scores
        for i in range(len(diff_peak)):
            if diff_peak[i] <= self.comp_set['pk_tolerance']:
                score_peaks_wavs[i] = 1 - diff_peak[i] / self.comp_set['pk_tolerance']
            else:
                score_peaks_wavs[i] = 0
        score_peaks_wav = np.mean(score_peaks_wavs)

        # Check fwhm of peaks
        diff_fwhm = np.full(spectra_compare.pks_num, self.comp_set['fwhm_tolerance'])

        for i in range(spectra_compare.pks_num):
            if matching_peak_index[i] != -1:
                diff_fwhm[i] = abs(spectra_compare.fwhm[i] - fluo_rec.fwhm[matching_peak_index[i]])

        score_fwhms = np.zeros(len(diff_fwhm), np.float_)

        for i in range(len(diff_fwhm)):
            if diff_fwhm[i] <= self.comp_set['fwhm_tolerance']:
                score_fwhms[i] = 1 - diff_fwhm[i] / self.comp_set['fwhm_tolerance']
            else:
                score_fwhms[i] = 0

        score_fwhm = np.mean(score_fwhms)

        score = score_num_peaks * self.comp_set['weight_peak'] * score_peaks_wav + self.comp_set['weight_fwhm'] * score_fwhm
        return score

    def compare_fluo_recs_shapes(self, spectra_compare_ints, spectra_compare_wavs, fluo_rec):
        sp_co_wavs, sp_co_ints = fluo_rec.get_spectra_dim_reduction(self.alpha)

        # In Avantes, there is always a direct correspondence between index and wavelength!
        # Remove spectra below a certain baseline
        indices_delete = spectra_compare_ints > self.comp_set['baseline_pts']
        spectra_compare_ints = np.delete(spectra_compare_ints, indices_delete)
        spectra_compare_wavs = np.delete(spectra_compare_wavs, indices_delete)
        sp_co_ints = np.delete(sp_co_ints, indices_delete)
        sp_co_wavs = np.delete(sp_co_wavs, indices_delete)

        scores = np.zeros(len(spectra_compare_ints), np.float_)

        for i in range(len(spectra_compare_ints)):
            diff = abs(spectra_compare_ints[i] - sp_co_ints[i])

            if diff <= self.comp_set['shape_pt_tol']:
                scores[i] = 1 - diff / self.comp_set['shape_pt_tol']
            else:
                scores[i] = 0

        score = np.mean(scores) * self.comp_set['weight_shape']
        return score

    def compare_spectra(self):
        # Conditioning Steps
        add_spectra_class = AddSpectra(self.led_range, self.smoothing_wind,
                                       self.min_peak_height, self.alpha, self.spectrum_to_read)
        (wavelengths_reduced, intensities_reduced, wavelengths, intensities, locations, fwhm, width_heights,
         left_ips, right_ips) = add_spectra_class.analyze_spectra()

        # Create fluorescence record for new spectra to compare
        spectra_compare = FluoRec(self.db_name, -1, "", len(locations), np.array(locations), np.array(fwhm), self.min_peak_height)

        # Comparison Part
        # Pre-allocate 100 records for algorithm speed purposes
        fluorescence_records = [FluoRec(self.db_name, -1, "", 0, np.array([0]), np.array([0]), 0) for _ in range(100)]
        found_peaks_cnt = 0

        # First Sorting of Database
        fs1_record_table = DB(self.db_name).get_db()
        idx = fs1_record_table['pks_num'] <= spectra_compare.pks_num
        fs1_record_table = fs1_record_table[idx]

        fs2_record_table = fs1_record_table.sort_values(by='id', ascending=False)

        # Peak Search
        added_ids = []
        for i in range(spectra_compare.pks_num):
            if found_peaks_cnt < 100:
                mask_greater = fs2_record_table['pks_wav'].apply(
                    lambda arr: any(x > spectra_compare.pks_wav[i] - self.comp_set['pk_tolerance'] for x in arr))
                mask_lower = fs2_record_table['pks_wav'].apply(
                    lambda arr: any(x < spectra_compare.pks_wav[i] + self.comp_set['pk_tolerance'] for x in arr))
                mask = mask_greater & mask_lower

                record_table = fs2_record_table[mask]
                height_table = len(record_table)

                for j in range(height_table):
                    if found_peaks_cnt < 100:
                        record = record_table.iloc[j, :]
                        if not record['id'] in added_ids:
                            fluorescence_records[found_peaks_cnt] = FluoRec(self.db_name, record['id'], record['Name'],
                                                                            record['pks_num'],
                                                                            record['pks_wav'],
                                                                            record['fwhm'],
                                                                            record['min_peak_height'])
                            added_ids.append(record['id'])
                            found_peaks_cnt += 1

        # Delete empty pre-allocated indices
        indices_to_delete = [i for i, rec in enumerate(fluorescence_records) if rec.id_ == -1]
        fluorescence_records = [rec for i, rec in enumerate(fluorescence_records) if i not in indices_to_delete]

        # Peak(s) Comparison
        # Assign Scores
        scores_table = pd.DataFrame(
            {'Score': np.zeros(len(fluorescence_records)), 'Index': np.arange(1, len(fluorescence_records) + 1)})
        for i in range(len(fluorescence_records)):
            peak_and_fwhm_score = self.compare_fluorescence_recs(spectra_compare, fluorescence_records[i])
            shape_score = self.compare_fluo_recs_shapes(intensities_reduced, wavelengths_reduced,
                                                fluorescence_records[i])
            scores_table.iloc[i, 0] = (peak_and_fwhm_score + shape_score) * 100
            scores_table.iloc[i, 1] = i

        # Sort the table in descending order
        sorted_score_table = scores_table.sort_values(by='Score', ascending=False)

        num_records_to_show = self.comp_set['num_records_to_show']
        if num_records_to_show < len(sorted_score_table):
            chosen_records_indices_to_show = sorted_score_table['Index'].iloc[:num_records_to_show]
            chosen_scores_to_show = sorted_score_table['Score'].iloc[:num_records_to_show]
        else:
            chosen_records_indices_to_show = sorted_score_table['Index']
            chosen_scores_to_show = sorted_score_table['Score']

        records = [FluoRec(self.db_name, -1, "", 0, np.array([0]), np.array([0]), 0) for _ in range(len(chosen_records_indices_to_show))]
        for index in chosen_records_indices_to_show:
            records[index] = fluorescence_records[int(index)]
            records[index].score = chosen_scores_to_show[index]

        # returns the records (as FluoRec class) and conditioned information about the original spectra
        return (records, wavelengths_reduced, intensities_reduced, wavelengths, intensities, locations, fwhm,
                width_heights, left_ips, right_ips)
