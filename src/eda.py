"""
Exploratory Data Analysis for Fraud Detection Project
Analyzes both fraud_data_final.csv and credit_data_final.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import traceback
from datetime import datetime
import warnings

# Configuration
warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8')  # Updated style name
sns.set_theme(style="whitegrid")  # Updated seaborn theme

def print_header(message):
    """Print formatted section header"""
    print("\n" + "=" * 60)
    print(f"=== {message.upper()} ===")
    print("=" * 60)

def log_message(message, level="INFO"):
    """Log formatted messages with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}][{level}] {message}")

def save_plot(fig, filename, plots_dir):
    """Save matplotlib figure to specified directory"""
    try:
        plots_dir.mkdir(exist_ok=True, parents=True)
        filepath = plots_dir / filename
        fig.savefig(filepath, bbox_inches='tight', dpi=150)
        plt.close(fig)
        log_message(f"Saved plot: {filename}")
        return True
    except Exception as e:
        log_message(f"Failed to save {filename}: {str(e)}", "ERROR")
        return False

def analyze_numeric_feature(df, feature, target, dataset_name, plots_dir):
    """Generate visualizations for numeric features"""
    try:
        if df[feature].nunique() <= 1:
            log_message(f"Skipping {feature} - no variance", "WARNING")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Distribution plot
        sns.histplot(data=df, x=feature, kde=True, ax=axes[0], color='skyblue')
        axes[0].set_title(f'Distribution of {feature}')
        
        # Boxplot by target
        sns.boxplot(data=df, x=target, y=feature, ax=axes[1], 
                   hue=target, palette='pastel', legend=False)
        axes[1].set_title(f'{feature} by {target}')
        
        plt.suptitle(f"{dataset_name}: {feature} Analysis", y=1.02)
        plt.tight_layout()
        save_plot(fig, f"{dataset_name}_{feature}_analysis.png", plots_dir)
        
    except Exception as e:
        log_message(f"Error analyzing {feature}: {str(e)}", "ERROR")

def analyze_categorical_feature(df, feature, target, dataset_name, plots_dir, max_categories=15):
    """Generate visualizations for categorical features"""
    try:
        unique_values = df[feature].nunique()
        if unique_values > max_categories:
            top_values = df[feature].value_counts().nlargest(max_categories).index
            plot_data = df[df[feature].isin(top_values)]
            title_suffix = f" (Top {max_categories})"
        else:
            plot_data = df
            title_suffix = ""

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Count plot
        sns.countplot(data=plot_data, x=feature, ax=axes[0], color='lightgreen')
        axes[0].set_title(f'Distribution of {feature}{title_suffix}')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Count plot by target
        sns.countplot(data=plot_data, x=feature, hue=target, 
                     ax=axes[1], palette='Set2')
        axes[1].set_title(f'{feature} by {target}')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.suptitle(f"{dataset_name}: {feature} Analysis", y=1.02)
        plt.tight_layout()
        save_plot(fig, f"{dataset_name}_{feature}_analysis.png", plots_dir)
        
    except Exception as e:
        log_message(f"Error analyzing {feature}: {str(e)}", "ERROR")

def analyze_dataset(df, dataset_name, target_col, plots_dir, important_features=None):
    """Main analysis function for a dataset"""
    print_header(f"analyzing {dataset_name}")
    
    # Basic dataset info
    log_message(f"Dataset shape: {df.shape}")
    log_message(f"Target distribution:\n{df[target_col].value_counts()}")
    
    # Determine features to analyze
    all_features = [col for col in df.columns if col != target_col]
    features_to_analyze = list(set(important_features or []) & set(all_features)) or all_features
    
    for feature in features_to_analyze:
        try:
            log_message(f"Analyzing feature: {feature}")
            
            if df[feature].dtype in ['int64', 'float64']:
                analyze_numeric_feature(df, feature, target_col, dataset_name, plots_dir)
            elif df[feature].dtype in ['object', 'category']:
                analyze_categorical_feature(df, feature, target_col, dataset_name, plots_dir)
            else:
                log_message(f"Skipping {feature} - unsupported dtype", "WARNING")
                
        except Exception as e:
            log_message(f"Failed to analyze {feature}: {str(e)}", "ERROR")

def main():
    """Main execution function"""
    print_header("starting exploratory data analysis")
    start_time = datetime.now()
    
    try:
        # Setup paths
        base_path = Path(__file__).parent.parent
        processed_dir = base_path / 'data/processed'
        plots_dir = base_path / 'notebooks/plots'
        
        # Load datasets
        log_message("Loading datasets...")
        fraud_data = pd.read_csv(processed_dir / 'fraud_data_final.csv')
        credit_data = pd.read_csv(processed_dir / 'credit_data_final.csv')
        
        # Define important features for each dataset
        fraud_important = [
            'purchase_value', 'age', 'country', 
            'hour_of_day', 'time_since_signup'
        ]
        
        credit_important = [
            'Amount', 'hour_of_day', 'day_of_week',
            'V1', 'V2', 'V3', 'V4', 'V5'
        ]
        
        # Analyze fraud data
        print_header("fraud data analysis")
        analyze_dataset(
            fraud_data,
            'fraud_data',
            target_col='class',
            plots_dir=plots_dir,
            important_features=fraud_important
        )
        
        # Analyze credit data
        print_header("credit data analysis")
        analyze_dataset(
            credit_data,
            'credit_data',
            target_col='Class',
            plots_dir=plots_dir,
            important_features=credit_important
        )
        
        # Completion
        runtime = (datetime.now() - start_time).total_seconds()
        print_header(f"analysis completed in {runtime:.2f} seconds")
        log_message(f"All plots saved to: {plots_dir}")
        
    except Exception as e:
        log_message(f"EDA failed: {str(e)}", "CRITICAL")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()