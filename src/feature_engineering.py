import pandas as pd
import numpy as np
from pathlib import Path
import sys
import traceback
from datetime import datetime

def print_header(message):
    """Print formatted section header"""
    print("\n" + "="*50)
    print(f"== {message}")
    print("="*50)

def log_message(message, level="INFO"):
    """Log formatted messages with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}][{level}] {message}")

def ip_to_integer(ip_address):
    """
    Convert IP address to integer - handles both string and numeric formats
    Args:
        ip_address: Could be string ("192.168.1.1") or numeric (3232235777.123)
    Returns:
        int: Integer representation of IP
    """
    try:
        # Handle numeric IPs (float or int)
        if isinstance(ip_address, (int, float, np.number)):
            return int(ip_address)  # Truncate decimal places
        
        # Handle string IPs
        if isinstance(ip_address, str):
            # If it's a string representation of a number
            if ip_address.replace('.', '', 1).isdigit():
                return int(float(ip_address))
            
            # If it's a traditional dotted IP string
            if ip_address.count('.') == 3:
                parts = list(map(int, ip_address.split('.')))
                return (parts[0] << 24) + (parts[1] << 16) + (parts[2] << 8) + parts[3]
        
        return None
    except Exception as e:
        log_message(f"IP conversion failed for {ip_address}: {str(e)}", "WARNING")
        return None

def merge_with_ip_data(fraud_df, ip_df):
    """
    Merge fraud data with IP country mapping using range matching
    Args:
        fraud_df (DataFrame): Processed fraud data
        ip_df (DataFrame): IP to country mapping data
    Returns:
        DataFrame: Merged dataframe with country information
    """
    log_message("Starting IP address to country mapping...")
    
    # Convert IPs to integers
    fraud_df['ip_integer'] = fraud_df['ip_address'].apply(ip_to_integer)
    
    # Report conversion issues
    failed_conversions = fraud_df['ip_integer'].isnull().sum()
    if failed_conversions > 0:
        bad_ips = fraud_df[fraud_df['ip_integer'].isna()]['ip_address'].unique()
        log_message(f"{failed_conversions} IP addresses failed conversion. Examples: {bad_ips[:5]}", "WARNING")
    
    # Prepare IP range data
    try:
        ip_df['lower_bound'] = ip_df['lower_bound_ip_address'].astype('float').astype('Int64')
        ip_df['upper_bound'] = ip_df['upper_bound_ip_address'].astype('float').astype('Int64')
        ip_df = ip_df.sort_values('lower_bound')
    except Exception as e:
        log_message(f"Failed to prepare IP ranges: {str(e)}", "ERROR")
        raise
    
    # Optimized country lookup
    def find_country(ip_int):
        if pd.isna(ip_int):
            return 'Unknown'
        try:
            matches = ip_df[(ip_df['lower_bound'] <= ip_int) & 
                          (ip_df['upper_bound'] >= ip_int)]
            return matches['country'].iloc[0] if not matches.empty else 'Unknown'
        except Exception:
            return 'Unknown'
    
    # Apply country mapping
    fraud_df['country'] = fraud_df['ip_integer'].apply(find_country)
    
    # Report matching statistics
    matched = (fraud_df['country'] != 'Unknown').sum()
    total = len(fraud_df)
    log_message(f"IP matching results: {matched:,} matched, {total-matched:,} unmatched ({matched/total:.1%} success rate)")
    
    return fraud_df

def create_time_features(df):
    """
    Create time-based features for fraud data
    Args:
        df (DataFrame): Merged fraud data
    Returns:
        DataFrame: Data with new time features
    """
    log_message("Creating time-based features...")
    
    # Ensure datetime conversion
    for col in ['signup_time', 'purchase_time']:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col])
    
    # Basic time features
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    
    # Time since signup (hours)
    df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600
    
    # User behavior features
    df = df.sort_values(['user_id', 'purchase_time'])
    
    # Transaction frequency
    user_freq = df.groupby('user_id').size().reset_index(name='user_transaction_freq')
    df = pd.merge(df, user_freq, on='user_id', how='left')
    
    # Time since last transaction
    df['time_since_last_txn'] = df.groupby('user_id')['purchase_time'].diff().dt.total_seconds() / 3600
    df['time_since_last_txn'] = df['time_since_last_txn'].fillna(0)  # First transaction
    
    # Purchase velocity (transactions per hour)
    df['purchase_velocity'] = df['user_transaction_freq'] / (df['time_since_signup'] + 1)  # +1 to avoid divide by zero
    
    log_message("Created time features: hour_of_day, day_of_week, time_since_signup, "
               "user_transaction_freq, time_since_last_txn, purchase_velocity")
    
    return df

def preprocess_credit_data(df):
    """
    Create features for credit card transaction data
    Args:
        df (DataFrame): Processed credit card data
    Returns:
        DataFrame: Data with new features
    """
    log_message("Processing credit card data features...")
    
    # Convert Time from seconds to datetime features
    df['hour_of_day'] = (df['Time'] / 3600) % 24
    df['day_of_week'] = (df['Time'] / (3600 * 24)) % 7
    
    # Transaction amount bins
    df['amount_bin'] = pd.cut(df['Amount'], 
                             bins=[0, 10, 50, 100, 500, 1000, float('inf')],
                             labels=['0-10', '10-50', '50-100', '100-500', '500-1000', '1000+'])
    
    log_message("Created credit card features: hour_of_day, day_of_week, amount_bin")
    
    return df

def save_dataset(df, filename, output_dir):
    """
    Save processed dataset to file
    Args:
        df (DataFrame): Data to save
        filename (str): Output filename
        output_dir (Path): Directory path
    """
    try:
        output_path = output_dir / filename
        df.to_csv(output_path, index=False)
        log_message(f"Saved dataset to {output_path}")
    except Exception as e:
        log_message(f"Failed to save {filename}: {str(e)}", "ERROR")
        raise

if __name__ == "__main__":
    print_header("FRAUD DETECTION FEATURE ENGINEERING PIPELINE")
    start_time = datetime.now()
    
    try:
        # Setup paths
        base_path = Path(__file__).parent.parent
        processed_dir = base_path / 'data/processed'
        processed_dir.mkdir(exist_ok=True)
        
        # Load data
        log_message("Loading processed datasets...")
        fraud_data = pd.read_csv(
            processed_dir / 'fraud_data_processed.csv',
            parse_dates=['signup_time', 'purchase_time']
        )
        ip_country = pd.read_csv(processed_dir / 'ip_country_processed.csv')
        credit_data = pd.read_csv(processed_dir / 'credit_data_processed.csv')
        log_message("Data loading completed")
        
        # Verify IP column format
        log_message(f"IP address column dtype: {fraud_data['ip_address'].dtype}")
        log_message(f"Sample IPs: {fraud_data['ip_address'].head().tolist()}")
        
        # Process fraud data
        print_header("FRAUD DATA FEATURE ENGINEERING")
        fraud_data_merged = merge_with_ip_data(fraud_data, ip_country)
        fraud_data_final = create_time_features(fraud_data_merged)
        
        # Validate output
        if fraud_data_final['ip_integer'].isna().all():
            raise ValueError("All IP conversions failed! Check IP format in input data")
            
        save_dataset(fraud_data_final, 'fraud_data_final.csv', processed_dir)
        
        # Process credit data
        print_header("CREDIT CARD DATA FEATURE ENGINEERING")
        credit_data_final = preprocess_credit_data(credit_data)
        save_dataset(credit_data_final, 'credit_data_final.csv', processed_dir)
        
        # Completion
        runtime = (datetime.now() - start_time).total_seconds()
        print_header(f"PIPELINE COMPLETED IN {runtime:.2f} SECONDS")
        log_message("Feature engineering finished successfully")
        
    except Exception as e:
        log_message(f"Pipeline failed: {str(e)}", "CRITICAL")
        traceback.print_exc()
        sys.exit(1)