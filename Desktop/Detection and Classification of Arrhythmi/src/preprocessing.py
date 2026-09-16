"""
ECG Signal Preprocessing and Heartbeat Segmentation Module.
Handles bandpass filtering, normalization, heartbeat segmentation around R-peaks,
AAMI class mapping, and dataset serialization to prevent data leakage.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

from src.data_loader import (
    discover_records, load_ecg_record, load_ecg_annotations,
    select_best_lead, BEAT_SYMBOLS, AAMI_CLASS_MAPPING, AAMI_CLASS_NAMES,
    PROCESSED_DATA_DIR
)


def butter_bandpass_filter(signal, fs=360.0, lowcut=0.5, highcut=45.0, order=2):
    """
    Applies a Butterworth bandpass filter to eliminate baseline wander and high-frequency noise.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


def normalize_heartbeat(segment):
    """
    Applies Min-Max normalization to scale heartbeat amplitudes to [0, 1].
    """
    min_val = np.min(segment)
    max_val = np.max(segment)
    if max_val - min_val > 0:
        return (segment - min_val) / (max_val - min_val)
    return segment


def segment_record_beats(rec, ann, record_id, window_before=90, window_after=90):
    """
    Extracts heartbeat segments around annotated R-peaks for a single record.
    
    Returns:
        List of dicts containing heartbeat metadata and normalized 180-sample vector.
    """
    raw_signal, channel_name = select_best_lead(rec)
    fs = rec.fs
    
    # 1. Bandpass filter signal
    filtered_signal = butter_bandpass_filter(raw_signal, fs=fs, lowcut=0.5, highcut=45.0, order=2)
    
    extracted_beats = []
    total_len = len(filtered_signal)
    
    for sample_idx, symbol in zip(ann.sample, ann.symbol):
        if symbol in BEAT_SYMBOLS:
            start_idx = sample_idx - window_before
            end_idx = sample_idx + window_after
            
            # Boundary check
            if start_idx >= 0 and end_idx < total_len:
                raw_segment = filtered_signal[start_idx:end_idx]
                norm_segment = normalize_heartbeat(raw_segment)
                aami_class = AAMI_CLASS_MAPPING.get(symbol, 'Q')
                
                beat_data = {
                    'record_id': str(record_id),
                    'sample_idx': int(sample_idx),
                    'lead_name': channel_name,
                    'raw_symbol': symbol,
                    'aami_class': aami_class,
                    'signal_vector': norm_segment.tolist()
                }
                extracted_beats.append(beat_data)
                
    return extracted_beats


def process_all_records(record_ids=None, data_dir=None, save_output=True):
    """
    Processes all valid WFDB records in the database, extracts heartbeats,
    and returns a structured DataFrame.
    """
    if record_ids is None:
        record_ids = discover_records(data_dir)
        
    all_beats = []
    symbol_counts = {}
    class_counts = {}
    
    print(f"Beginning preprocessing for {len(record_ids)} MIT-BIH records...")
    
    for idx, rec_id in enumerate(record_ids):
        try:
            rec = load_ecg_record(rec_id, data_dir)
            ann = load_ecg_annotations(rec_id, data_dir)
            beats = segment_record_beats(rec, ann, rec_id)
            all_beats.extend(beats)
            
            for b in beats:
                sym = b['raw_symbol']
                cls = b['aami_class']
                symbol_counts[sym] = symbol_counts.get(sym, 0) + 1
                class_counts[cls] = class_counts.get(cls, 0) + 1
                
        except Exception as e:
            print(f"Warning: Failed to process record {rec_id}: {e}")

    # Build DataFrame
    rows = []
    for b in all_beats:
        row = {
            'record_id': b['record_id'],
            'sample_idx': b['sample_idx'],
            'lead_name': b['lead_name'],
            'raw_symbol': b['raw_symbol'],
            'aami_class': b['aami_class']
        }
        # Expand 180 signal samples as features s_0 to s_179
        for i, val in enumerate(b['signal_vector']):
            row[f's_{i}'] = val
        rows.append(row)
        
    df = pd.DataFrame(rows)
    
    print(f"\nPreprocessing Complete!")
    print(f"Total Heartbeats Extracted: {len(df)}")
    print(f"Extracted Sample Dimension: {180} amplitude points per heartbeat")
    print(f"AAMI Class Distribution: {class_counts}")
    
    if save_output:
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        csv_path = os.path.join(PROCESSED_DATA_DIR, "processed_beats.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved processed dataset to: {csv_path}")
        
        # Save summary JSON
        summary_info = {
            'total_records': len(record_ids),
            'valid_records': record_ids,
            'total_heartbeats': len(df),
            'sampling_rate_hz': 360,
            'heartbeat_window_samples': 180,
            'raw_symbol_counts': symbol_counts,
            'aami_class_counts': class_counts,
            'aami_class_names': AAMI_CLASS_NAMES
        }
        json_path = os.path.join(PROCESSED_DATA_DIR, "dataset_summary.json")
        with open(json_path, 'w') as f:
            json.dump(summary_info, f, indent=2)
            
    return df
