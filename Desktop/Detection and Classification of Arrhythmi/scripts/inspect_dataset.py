"""
Dataset Inspection Script for ECG Arrhythmia Detection Project.
Scans data/raw/ and data/processed/ for available datasets, validates structure,
checks missing values, class labels, and prints a comprehensive inspection report.
"""

import sys
import os

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.data_loader import find_dataset_files, load_raw_csv_dataset, get_dataset_summary


def inspect_dataset():
    print("=" * 60)
    print("      ECG ARRHYTHMIA DATASET INSPECTION TOOL")
    print("=" * 60)

    # 1. Scan data/raw/
    raw_info = find_dataset_files()
    print(f"\nScanning raw data directory: {raw_info['directory']}")
    print(f"Found CSV files: {len(raw_info['csv_files'])}")
    print(f"Found WFDB .dat files: {len(raw_info['dat_files'])}")
    print(f"Found WFDB .hea header files: {len(raw_info['hea_files'])}")

    if raw_info['total_files'] == 0:
        print("\n[STATUS]: DATASET NOT PRESENT IN data/raw/")
        print("-" * 60)
        print("INSTRUCTIONS TO DOWNLOAD & PLACE MIT-BIH DATASET:")
        print("1. Official PhysioNet MIT-BIH Arrhythmia Database:")
        print("   URL: https://physionet.org/content/mitdb/1.0.0/")
        print("2. Standard Preprocessed Kaggle CSV Version (MIT-BIH Arrhythmia Dataset):")
        print("   URL: https://www.kaggle.com/datasets/shayanfazeli/heartbeat")
        print("   Files: mitbih_train.csv and mitbih_test.csv")
        print("3. Alternatively, place your CSV dataset (e.g., mitbih_train.csv, mitbih_database.csv,")
        print("   or arrhythmia.csv) into the directory:")
        print("   -> data/raw/")
        print("=" * 60)
        return False

    # 2. Inspect first CSV file found
    if raw_info['csv_files']:
        file_path = raw_info['csv_files'][0]
        print(f"\n[INSPECTING FILE]: {os.path.basename(file_path)}")
        try:
            df, loaded_path = load_raw_csv_dataset(file_path)
            print(f"Loaded DataFrame Shape: {df.shape} (Rows: {df.shape[0]}, Columns: {df.shape[1]})")

            # Determine target column (usually last column)
            target_col = df.columns[-1]
            summary = get_dataset_summary(df, target_col=target_col)

            print(f"Target Column Identified: '{target_col}'")
            print(f"Total Samples: {summary['num_samples']}")
            print(f"Feature Count: {summary['num_features'] - 1}")
            print(f"Missing Values: {summary['missing_values_count']}")
            print(f"Duplicate Rows: {summary['duplicate_rows_count']}")
            print(f"Number of Detected Classes: {summary.get('num_classes', 'N/A')}")
            
            if 'class_distribution' in summary:
                print("\nClass Distribution:")
                for cls, count in summary['class_distribution'].items():
                    print(f"  Class {cls}: {count} samples")

        except Exception as e:
            print(f"Error inspecting dataset file: {e}")
            return False

    return True


if __name__ == "__main__":
    inspect_dataset()
