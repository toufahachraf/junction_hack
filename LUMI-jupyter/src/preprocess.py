import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import argparse

"""
preprocess.py — LUMI / Local / Jupyter compatible
--------------------------------------------------
No GCP dependency. Data is read from and written to local paths.

Usage:
    python preprocess.py --input /scratch/project_xxx/HI-Small_Trans.csv

On LUMI, store your CSV on the scratch filesystem and point --input at it.
On a local Jupyter environment, point --input at wherever you downloaded the dataset.
"""

def main():
    parser = argparse.ArgumentParser(description="QAML Data Preprocessor")
    parser.add_argument(
        "--input",
        type=str,
        default="data/HI-Small_Trans.csv",
        help="Path to the raw HI-Small_Trans.csv file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to write processed CSVs into"
    )
    args = parser.parse_args()

    print(f"Loading dataset from: {args.input}")
    df = pd.read_csv(args.input)

    # Drop duplicate columns (pandas sometimes creates these on re-read)
    df = df.loc[:, ~df.columns.duplicated()]

    print("Dropping identifier columns...")
    cols_to_drop = ['Timestamp', 'Account', 'Account.1', 'To Bank']
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # One-hot encode categorical features
    print("Applying One-Hot Encoding...")
    cat_cols = ['Payment Format', 'Payment Currency', 'Receiving Currency']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    df_encoded = pd.get_dummies(df, columns=[c for c in cat_cols if c in df.columns])

    # Convert bool dummy columns to int
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'bool':
            df_encoded[col] = df_encoded[col].astype(int)

    # Log-transform and scale numerical amounts to [-pi, pi] for AngleEmbedding
    print("Applying Log-Transform and normalizing numerical amounts to [-pi, pi]...")
    num_cols = ['Amount Paid', 'Amount Received']
    scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
    for col in num_cols:
        if col in df_encoded.columns:
            df_encoded[col] = np.log1p(df_encoded[col])
            df_encoded[col] = scaler.fit_transform(df_encoded[[col]])

    print("Splitting datasets for Federated Learning...")
    if 'From Bank' in df_encoded.columns:
        df_encoded['From Bank'] = df_encoded['From Bank'].astype(str)
        bank_70_train = df_encoded[
            (df_encoded['From Bank'] == '70') & (df_encoded['Is Laundering'] == 0)
        ].copy()
        bank_10_train = df_encoded[
            (df_encoded['From Bank'] == '10') & (df_encoded['Is Laundering'] == 0)
        ].copy()
    else:
        print("Warning: 'From Bank' column not found — falling back to random 50/50 split.")
        normal_data = df_encoded[df_encoded['Is Laundering'] == 0].sample(frac=1, random_state=42)
        mid = len(normal_data) // 2
        bank_70_train = normal_data.iloc[:mid].copy()
        bank_10_train = normal_data.iloc[mid:].copy()

    # Build global test set: all illicit + 10x normal sample
    illicit_df = df_encoded[df_encoded['Is Laundering'] == 1]
    normal_sample = df_encoded[df_encoded['Is Laundering'] == 0].sample(
        n=len(illicit_df) * 10, random_state=42
    )
    global_test = pd.concat([illicit_df, normal_sample]).sample(frac=1, random_state=42)

    # Drop From Bank from all splits
    for split in [bank_70_train, bank_10_train, global_test]:
        if 'From Bank' in split.columns:
            split.drop(columns=['From Bank'], inplace=True)

    # Drop Is Laundering from training sets (unsupervised)
    for split in [bank_70_train, bank_10_train]:
        if 'Is Laundering' in split.columns:
            split.drop(columns=['Is Laundering'], inplace=True)

    os.makedirs(args.output_dir, exist_ok=True)
    bank_70_train.to_csv(f"{args.output_dir}/bank_70_train.csv", index=False)
    bank_10_train.to_csv(f"{args.output_dir}/bank_10_train.csv", index=False)
    global_test.to_csv(f"{args.output_dir}/global_test.csv", index=False)

    print("\nData processed successfully.")
    print(f"Final Quantum Feature Count : {len(bank_70_train.columns)}")
    print(f"Bank 70 Training Samples   : {len(bank_70_train)}")
    print(f"Bank 10 Training Samples   : {len(bank_10_train)}")
    print(f"Global Test Samples        : {len(global_test)}")
    print(f"\nOutputs written to: {args.output_dir}/")


if __name__ == "__main__":
    main()
