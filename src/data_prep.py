import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from google.cloud import storage
import os

def main():
    print("Loading dataset...")
    df = pd.read_csv('gs://junction-aml-quantum-bucket/HI-Small_Trans.csv')

    df = df.loc[:, ~df.columns.duplicated()] 
    cols_to_drop = ['Timestamp', 'Account', 'Account.1', 'To Bank']
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    cat_cols = ['Payment Format', 'Payment Currency', 'Receiving Currency']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    df_encoded = pd.get_dummies(df, columns=[c for c in cat_cols if c in df.columns])

    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'bool':
            df_encoded[col] = df_encoded[col].astype(int)

    num_cols = ['Amount Paid', 'Amount Received']
    scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
    for col in num_cols:
        if col in df_encoded.columns:
            df_encoded[col] = np.log1p(df_encoded[col])
            df_encoded[col] = scaler.fit_transform(df_encoded[[col]])

    if 'From Bank' in df_encoded.columns:
        df_encoded['From Bank'] = df_encoded['From Bank'].astype(str)
        bank_70_train = df_encoded[(df_encoded['From Bank'] == '70') & (df_encoded['Is Laundering'] == 0)].copy()
        bank_10_train = df_encoded[(df_encoded['From Bank'] == '10') & (df_encoded['Is Laundering'] == 0)].copy()
    else:
        normal_data = df_encoded[df_encoded['Is Laundering'] == 0].sample(frac=1)
        mid = len(normal_data) // 2
        bank_70_train = normal_data.iloc[:mid].copy()
        bank_10_train = normal_data.iloc[mid:].copy()

    illicit_df = df_encoded[df_encoded['Is Laundering'] == 1]
    normal_sample = df_encoded[df_encoded['Is Laundering'] == 0].sample(n=len(illicit_df)*10, random_state=42)
    global_test = pd.concat([illicit_df, normal_sample]).sample(frac=1, random_state=42)

    if 'From Bank' in bank_70_train.columns:
        bank_70_train.drop(columns=['From Bank'], inplace=True)
        bank_10_train.drop(columns=['From Bank'], inplace=True)
        global_test.drop(columns=['From Bank'], inplace=True)
        
    if 'Is Laundering' in bank_70_train.columns:
        bank_70_train.drop(columns=['Is Laundering'], inplace=True)
        bank_10_train.drop(columns=['Is Laundering'], inplace=True)

    os.makedirs('data/processed', exist_ok=True)
    bank_70_train.to_csv('data/processed/bank_70_train.csv', index=False)
    bank_10_train.to_csv('data/processed/bank_10_train.csv', index=False)
    global_test.to_csv('data/processed/global_test.csv', index=False)

    print(f"Data processed successfully.")
    print(f"Features: {len(bank_70_train.columns)}")
    print(f"Bank 70: {len(bank_70_train)} | Bank 10: {len(bank_10_train)}")
    
    bucket_name = "junction-aml-quantum-bucket"
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        for file_name in ['bank_70_train.csv', 'bank_10_train.csv', 'global_test.csv']:
            local_path = f'data/processed/{file_name}'
            blob = bucket.blob(file_name)
            blob.upload_from_filename(local_path)
            
    except Exception as e:
        print(f"GCP Upload failed: {e}")

if __name__ == "__main__":
    main()
