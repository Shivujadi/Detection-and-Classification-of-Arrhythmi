"""
Test and Pipeline Execution Script for Phase 4.
Executes end-to-end WFDB record discovery, heartbeat segmentation,
preprocessing, dataset serialization, and plot generation.
"""

import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.data_loader import discover_records, load_ecg_record, load_ecg_annotations
from src.preprocessing import process_all_records
from scripts.generate_plots import generate_ecg_plots


def main():
    print("=" * 65)
    print("      PHASE 4: ECG DATA PREPROCESSING & SEGMENTATION TEST")
    print("=" * 65)

    # 1. Discover WFDB records
    records = discover_records()
    print(f"\n[STEP 1]: Discovered {len(records)} valid WFDB records in data/raw/mitdb/")
    if len(records) == 0:
        print("ERROR: No valid WFDB records found!")
        sys.exit(1)
        
    print(f"Record list sample: {records[:10]} ...")

    # 2. Test single record load
    test_rec_id = records[0]
    rec = load_ecg_record(test_rec_id)
    ann = load_ecg_annotations(test_rec_id)
    print(f"\n[STEP 2]: Loaded Record {test_rec_id}")
    print(f"  - Sampling Frequency: {rec.fs} Hz")
    print(f"  - Channels: {rec.sig_name}")
    print(f"  - Total Record Length: {len(rec.p_signal)} samples ({len(rec.p_signal)/rec.fs:.1f} seconds)")
    print(f"  - Annotations Count: {len(ann.sample)}")

    # 3. Process all records
    print(f"\n[STEP 3]: Processing and Segmenting Heartbeats across all {len(records)} records...")
    df = process_all_records(record_ids=records, save_output=True)

    # 4. Verify outputs
    processed_csv = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "processed_beats.csv")
    summary_json = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "dataset_summary.json")

    assert os.path.exists(processed_csv), f"Error: {processed_csv} missing!"
    assert os.path.exists(summary_json), f"Error: {summary_json} missing!"
    print("\n[STEP 4]: Verified output serialization files:")
    print(f"  - {processed_csv} (Size: {os.path.getsize(processed_csv)/1e6:.2f} MB)")
    print(f"  - {summary_json}")

    # 5. Generate plots
    print(f"\n[STEP 5]: Generating ECG Plots...")
    generate_ecg_plots()

    print("\n" + "=" * 65)
    print("  SUCCESS: Phase 4 ECG Preprocessing & Validation Pipeline Passed!")
    print("=" * 65)


if __name__ == "__main__":
    main()
