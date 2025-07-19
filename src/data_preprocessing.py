import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def print_header(message):
    print("\n" + "="*50)
    print(f"== {message}")
    print("="*50)

def load_data():
    """Load all raw datasets"""
    try:
        base_path = Path(__file__).parent.parent
        print("\nLoading datasets...")
        
        fraud_data = pd.read_csv(base_path / 'data/raw/Fraud_Data.csv')
        print("[SUCCESS] Fraud_Data.csv loaded")
        
        ip_country = pd.read_csv(base_path / 'data/raw/IpAddress_to_Country.csv')
        print("[SUCCESS] IpAddress_to_Country.csv loaded")
        
        credit_data = pd.read_csv(base_path / 'data/raw/creditcard.csv')
        print("[SUCCESS] creditcard.csv loaded")
        
        return fraud_data, ip_country, credit_data
        
    except Exception as e:
        print(f"\n[ERROR] Failed to load data: {str(e)}")
        sys.exit(1)

def handle_missing_values(df, df_name):
    """Handle missing values in dataframe"""
    print(f"\nHandling missing values for {df_name}:")
    print("-"*40)
    
    # Check initial missing values
    missing_before = df.isnull().sum().sum()
    print(f"Total missing values before: {missing_before}")
    
    if missing_before > 0:
        # Fill numerical missing values with median
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in num_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical missing values with mode
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        # Verify
        missing_after = df.isnull().sum().sum()
        print(f"Total missing values after: {missing_after}")
        
        if missing_after > 0:
            print("[WARNING] Some missing values remain")
        else:
            print("[SUCCESS] All missing values handled")
    else:
        print("[INFO] No missing values found")
    
    return df

def remove_duplicates(df, df_name):
    """Remove duplicate rows from dataframe"""
    print(f"\nRemoving duplicates for {df_name}:")
    print("-"*40)
    
    initial_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    final_rows = df.shape[0]
    duplicates_removed = initial_rows - final_rows
    
    print(f"Initial rows: {initial_rows}")
    print(f"Final rows: {final_rows}")
    print(f"Duplicates removed: {duplicates_removed}")
    
    if duplicates_removed > 0:
        print("[SUCCESS] Duplicates removed")
    else:
        print("[INFO] No duplicates found")
    
    return df

def correct_data_types(df, df_name):
    """Convert columns to proper data types"""
    print(f"\nCorrecting data types for {df_name}:")
    print("-"*40)
    
    changes_made = False
    
    if 'signup_time' in df.columns:
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        print("Converted 'signup_time' to datetime")
        changes_made = True
    
    if 'purchase_time' in df.columns:
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        print("Converted 'purchase_time' to datetime")
        changes_made = True
    
    if not changes_made:
        print("[INFO] No data type conversions needed")
    else:
        print("[SUCCESS] Data types corrected")
    
    return df

def save_processed_data(df, filename):
    """Save processed dataframe to file"""
    try:
        processed_path = Path(__file__).parent.parent / 'data/processed'
        processed_path.mkdir(exist_ok=True)
        
        filepath = processed_path / filename
        df.to_csv(filepath, index=False)
        print(f"[SUCCESS] Saved processed data to {filepath}")
    except Exception as e:
        print(f"[ERROR] Failed to save {filename}: {str(e)}")

if __name__ == "__main__":
    print_header("STARTING DATA PREPROCESSING PIPELINE")
    
    try:
        # Load all datasets
        fraud_data, ip_country, credit_data = load_data()
        
        # Process Fraud Data
        print_header("PROCESSING FRAUD DATA")
        fraud_data = handle_missing_values(fraud_data, "Fraud Data")
        fraud_data = remove_duplicates(fraud_data, "Fraud Data")
        fraud_data = correct_data_types(fraud_data, "Fraud Data")
        save_processed_data(fraud_data, "fraud_data_processed.csv")
        
        # Process Credit Data
        print_header("PROCESSING CREDIT CARD DATA")
        credit_data = handle_missing_values(credit_data, "Credit Data")
        credit_data = remove_duplicates(credit_data, "Credit Data")
        save_processed_data(credit_data, "credit_data_processed.csv")
        
        # Process IP Country Data
        print_header("PROCESSING IP COUNTRY DATA")
        ip_country = handle_missing_values(ip_country, "IP Country Data")
        ip_country = remove_duplicates(ip_country, "IP Country Data")
        save_processed_data(ip_country, "ip_country_processed.csv")
        
        print_header("PREPROCESSING COMPLETED SUCCESSFULLY")
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Pipeline failed: {str(e)}")
        sys.exit(1)