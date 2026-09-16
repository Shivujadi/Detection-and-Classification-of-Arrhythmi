"""
ECG Visualization Plot Generator.
Generates and saves publication-quality ECG plot figures under static/plots/:
1. Raw ECG Signal Waveform (Record 100 Lead MLII) with R-peak annotations.
2. Segmented Heartbeat Samples for each AAMI Arrhythmia Class.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.data_loader import load_ecg_record, load_ecg_annotations, select_best_lead, AAMI_CLASS_NAMES
from src.preprocessing import butter_bandpass_filter, normalize_heartbeat, BEAT_SYMBOLS, AAMI_CLASS_MAPPING


PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "plots")


def generate_ecg_plots():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # -------------------------------------------------------------
    # 1. Plot Raw & Filtered Continuous ECG Signal (Record 100)
    # -------------------------------------------------------------
    print("Generating Raw ECG Signal Waveform Plot...")
    rec = load_ecg_record('100')
    ann = load_ecg_annotations('100')
    raw_sig, lead_name = select_best_lead(rec)
    filtered_sig = butter_bandpass_filter(raw_sig, fs=rec.fs)
    
    # Take 5 seconds slice (1800 samples)
    start_sample = 1000
    end_sample = start_sample + 1800
    time_sec = np.arange(1800) / rec.fs
    
    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
    ax.plot(time_sec, raw_sig[start_sample:end_sample], color='#94a3b8', alpha=0.7, label='Raw Signal')
    ax.plot(time_sec, filtered_sig[start_sample:end_sample], color='#2563eb', linewidth=1.5, label='Filtered ECG (MLII)')
    
    # Overlay R-peak annotations in slice
    ann_in_slice = [(s - start_sample, sym) for s, sym in zip(ann.sample, ann.symbol) if start_sample <= s < end_sample]
    for rel_sample, sym in ann_in_slice:
        t_val = rel_sample / rec.fs
        y_val = filtered_sig[start_sample + rel_sample]
        ax.plot(t_val, y_val, 'ro', markersize=6)
        ax.annotate(f"{sym}", (t_val, y_val + 0.15), ha='center', fontsize=10, fontweight='bold', color='#dc2626')
        
    ax.set_title('MIT-BIH Record 100 — Continuous ECG Waveform & R-Peak Annotations', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Time (seconds)', fontsize=10)
    ax.set_ylabel('Amplitude (mV)', fontsize=10)
    ax.legend(loc='upper right')
    plt.tight_layout()
    
    raw_plot_path = os.path.join(PLOTS_DIR, "raw_ecg_waveform.png")
    fig.savefig(raw_plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {raw_plot_path}")

    # -------------------------------------------------------------
    # 2. Plot Segmented Heartbeats for Each AAMI Class
    # -------------------------------------------------------------
    print("Generating Segmented Heartbeats Plot across AAMI Classes...")
    processed_csv = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "processed_beats.csv")
    
    if os.path.exists(processed_csv):
        df = pd.read_csv(processed_csv)
    else:
        from src.preprocessing import process_all_records
        df = process_all_records(save_output=True)
        
    signal_cols = [c for c in df.columns if c.startswith('s_')]
    classes = sorted(df['aami_class'].unique())
    
    fig, axes = plt.subplots(len(classes), 1, figsize=(10, 2.5 * len(classes)), sharex=True, dpi=150)
    if len(classes) == 1:
        axes = [axes]
        
    time_ms = (np.arange(180) - 90) * (1000 / 360)  # centered around 0 ms
    
    colors = {'N': '#16a34a', 'S': '#ea580c', 'V': '#dc2626', 'F': '#9333ea', 'Q': '#0284c7'}
    
    for ax, cls in zip(axes, classes):
        class_beats = df[df['aami_class'] == cls][signal_cols].values
        n_samples = min(50, len(class_beats))
        
        # Plot individual faint beats
        for i in range(n_samples):
            ax.plot(time_ms, class_beats[i], color=colors.get(cls, '#64748b'), alpha=0.15, linewidth=0.8)
            
        # Plot mean beat waveform
        mean_beat = np.mean(class_beats, axis=0)
        ax.plot(time_ms, mean_beat, color=colors.get(cls, '#0f172a'), linewidth=2.5, label=f"Mean Beat ({cls})")
        
        cls_fullname = AAMI_CLASS_NAMES.get(cls, cls)
        ax.set_title(f"AAMI Class: {cls_fullname} (Count: {len(class_beats)})", fontsize=11, fontweight='bold')
        ax.set_ylabel('Norm. Amp', fontsize=9)
        ax.axvline(0, color='#ef4444', linestyle='--', alpha=0.6, label='R-Peak Center')
        ax.legend(loc='upper right', fontsize=8)
        
    axes[-1].set_xlabel('Time relative to R-peak (ms)', fontsize=10)
    plt.tight_layout()
    
    beats_plot_path = os.path.join(PLOTS_DIR, "segmented_heartbeats.png")
    fig.savefig(beats_plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {beats_plot_path}")


if __name__ == "__main__":
    generate_ecg_plots()
