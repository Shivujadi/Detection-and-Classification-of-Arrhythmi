"""
Data Loader Module for PhysioNet MIT-BIH Arrhythmia WFDB Dataset.
Provides functions for discovering valid WFDB records, reading signals,
loading annotations, and retrieving metadata.
"""

import os
import glob
import wfdb
import numpy as np
import pandas as pd


RAW_MITDB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "mitdb")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")

# Recognized AAMI Beat Annotation Symbols
BEAT_SYMBOLS = {'N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F', '/', 'f', 'Q'}

# AAMI 5-Class Standard Mapping
AAMI_CLASS_MAPPING = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',  # Normal / Bundle Branch Block
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',             # Supraventricular Ectopic Beat (SVEB)
    'V': 'V', 'E': 'V',                                 # Ventricular Ectopic Beat (VEB)
    'F': 'F',                                           # Fusion Beat
    '/': 'Q', 'f': 'Q', 'Q': 'Q'                        # Paced / Unknown / Unclassifiable (Q)
}

AAMI_CLASS_NAMES = {
    'N': 'Normal / Bundle Branch Block (N)',
    'S': 'Supraventricular Ectopic Beat (S)',
    'V': 'Ventricular Ectopic Beat (V)',
    'F': 'Fusion Beat (F)',
    'Q': 'Paced / Unknown / Unclassifiable (Q)'
}


def get_mitdb_dir(data_dir=None):
    """
    Returns the valid path to MIT-BIH database folder.
    """
    if data_dir is None:
        data_dir = RAW_MITDB_DIR
    if not os.path.exists(data_dir):
        # Fallback check for parent raw dir
        parent_dir = os.path.dirname(data_dir)
        if os.path.exists(os.path.join(parent_dir, "mitdb")):
            return os.path.join(parent_dir, "mitdb")
    return data_dir


def discover_records(data_dir=None):
    """
    Scans the data directory and returns a sorted list of valid record IDs
    that have complete .hea, .dat, and .atr files.
    """
    db_dir = get_mitdb_dir(data_dir)
    if not os.path.exists(db_dir):
        raise FileNotFoundError(f"MIT-BIH directory not found at: {db_dir}")

    records_file = os.path.join(db_dir, "RECORDS")
    candidate_records = []

    if os.path.exists(records_file):
        with open(records_file, "r") as f:
            candidate_records = [line.strip() for line in f if line.strip()]
    else:
        hea_files = glob.glob(os.path.join(db_dir, "*.hea"))
        candidate_records = sorted(list(set([os.path.splitext(os.path.basename(f))[0] for f in hea_files])))

    valid_records = []
    for rec_id in candidate_records:
        rec_path = os.path.join(db_dir, rec_id)
        if (os.path.exists(rec_path + ".hea") and 
            os.path.exists(rec_path + ".dat") and 
            os.path.exists(rec_path + ".atr")):
            valid_records.append(rec_id)

    return valid_records


def load_ecg_record(record_id, data_dir=None):
    """
    Loads a single WFDB record signal and header metadata.
    
    Returns:
        rec (wfdb.Record): Loaded record object containing p_signal, sig_name, fs, etc.
    """
    db_dir = get_mitdb_dir(data_dir)
    rec_path = os.path.join(db_dir, str(record_id))
    
    if not (os.path.exists(rec_path + ".hea") and os.path.exists(rec_path + ".dat")):
        raise FileNotFoundError(f"Record files for '{record_id}' missing in {db_dir}.")

    rec = wfdb.rdrecord(rec_path)
    return rec


def load_ecg_annotations(record_id, data_dir=None):
    """
    Loads beat annotations for a given WFDB record.
    
    Returns:
        ann (wfdb.Annotation): Annotation object containing sample indices and symbols.
    """
    db_dir = get_mitdb_dir(data_dir)
    rec_path = os.path.join(db_dir, str(record_id))
    
    if not os.path.exists(rec_path + ".atr"):
        raise FileNotFoundError(f"Annotation file '.atr' for '{record_id}' missing in {db_dir}.")

    ann = wfdb.rdann(rec_path, "atr")
    return ann


def select_best_lead(rec):
    """
    Selects the best ECG lead signal (preferring MLII if available, else first channel).
    
    Returns:
        signal (np.ndarray): 1D numpy array of the selected ECG lead.
        channel_name (str): Name of the selected channel.
    """
    sig_names = rec.sig_name
    target_lead = 'MLII'
    
    if target_lead in sig_names:
        ch_idx = sig_names.index(target_lead)
    else:
        ch_idx = 0  # Fallback to first available channel
        
    return rec.p_signal[:, ch_idx], sig_names[ch_idx]
