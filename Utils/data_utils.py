import datetime
import pandas as pd
import numpy as np
import scipy.sparse

from sklearn.metrics import roc_auc_score

def window_data(
    window_lengths = None,
    feature_matrix = None,
    all_feature_names = None,
    cohort = None ,
    featureSet = None,
    cohort_end_date_col='training_end_date'
):
    def gen_window_ixs(window_days):
        window_start = pd.to_datetime(
            cohort._cohort_generation_kwargs[cohort_end_date_col]
        ) - datetime.timedelta(days = window_days)
        window_end = pd.to_datetime(cohort._cohort_generation_kwargs[cohort_end_date_col])
        windowed_rev_time_map = {
            t:i for i,t in featureSet.time_map.items()
            if pd.to_datetime(t) < window_end and pd.to_datetime(t) > window_start
        }
        return list(windowed_rev_time_map.values())
    windowed_time_ixs = dict()
    for interval in window_lengths:
        windowed_time_ixs[interval] = gen_window_ixs(interval)
    feature_matrix_slices = []
    feature_names = []
    for interval in sorted(windowed_time_ixs):
        feature_matrix_slices.append(
            feature_matrix[:, windowed_time_ixs[interval], :]
        )
        feature_names += [
            '{} - {} days'.format(n, interval)
            for i,n in enumerate(all_feature_names)
        ]
    feature_matrix_counts = scipy.sparse.vstack(
        [
            m.sum(axis=1).T.tocsr()
            for m in feature_matrix_slices
        ]
    )
    return feature_matrix_counts, feature_names


def window_data_sorted(
    window_lengths = None,
    feature_matrix = None,
    all_feature_names = None,
    cohort = None ,
    featureSet = None,
    cohort_end_date_col='training_end_date'
):
    all_times = pd.to_datetime(
        np.array(list(featureSet.time_map.values()))
    )
    
    def gen_window_ixs(window_days):
        
        window_start = pd.to_datetime(
            cohort._cohort_generation_kwargs[cohort_end_date_col]
        ) - datetime.timedelta(days = window_days)
        window_end = pd.to_datetime(cohort._cohort_generation_kwargs[cohort_end_date_col])
        return all_times.searchsorted(window_start), all_times.searchsorted(window_end)
            
    windowed_time_ixs = dict()
    for interval in window_lengths:
        windowed_time_ixs[interval] = gen_window_ixs(interval)
    feature_matrix_slices = []
    feature_names = []
    for interval in sorted(windowed_time_ixs):
        feature_matrix_slices.append(
            feature_matrix[
                :, windowed_time_ixs[interval][0]:windowed_time_ixs[interval][1], :
            ]
        )
        feature_names += [
            '{} - {} days'.format(n, interval)
            for i,n in enumerate(all_feature_names)
        ]
    feature_matrix_counts = scipy.sparse.vstack(
        [
            m.sum(axis=1).T.tocsr()
            for m in feature_matrix_slices
        ]
    )
    return feature_matrix_counts, feature_names


def auc_std(true, pred):
    auc = roc_auc_score(true, pred)
    n1 = sum(true)
    n2 = len(true) - sum(true)
    auc_var = (
        auc * (1 - auc) +
        (n1 - 1) * ((
            auc / (2 - auc)
        ) - auc * auc) +
        (n2 - 1) * ((
            2 * auc * auc / (1 + auc)
        ) - auc * auc)
    ) / (n1 * n2)
    return auc_var ** 0.5