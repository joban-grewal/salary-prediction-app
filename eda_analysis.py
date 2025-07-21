import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# Load your dataset
try:
    df = pd.read_csv("data.csv")
    print("✅ Dataset loaded successfully!")
except FileNotFoundError:
    print("❌ Error: 'data.csv' not found. Please ensure your dataset file exists.")
    exit(1)

# --- 1. Basic Overview ---
print("=" * 60)
print("📊 BASIC DATASET OVERVIEW")
print("=" * 60)
print(f"Shape of dataset: {df.shape}")
print(f"Number of rows: {df.shape[0]:,}")
print(f"Number of columns: {df.shape[1]}")

print("\n📋 Column Information:")
print(df.dtypes)

print("\n🔍 Missing Values:")
missing_data = df.isnull().sum()
missing_percentage = (missing_data / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing_data,
    'Percentage': missing_percentage
}).sort_values('Missing Count', ascending=False)
print(missing_df[missing_df['Missing Count'] > 0])

print(f"\n🔄 Duplicates: {df.duplicated().sum()}")

# --- 2. Descriptive Statistics ---
print("\n" + "=" * 60)
print("📈 DESCRIPTIVE STATISTICS")
print("=" * 60)
print("\nNumerical columns summary:")
numerical_cols = df.select_dtypes(include=[np.number]).columns
if len(numerical_cols) > 0:
    print(df[numerical_cols].describe())
else:
    print("No numerical columns found")

print("\nCategorical columns summary:")
categorical_cols = df.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    for col in categorical_cols:
        print(f"\n{col}:")
        value_counts = df[col].value_counts().head()
        print(value_counts)
else:
    print("No categorical columns found")

# --- 3. Univariate Analysis (Histograms) ---
print("\n" + "=" * 60)
print("📊 CREATING VISUALIZATIONS")
print("=" * 60)

if len(numerical_cols) > 0:
    # Calculate subplot dimensions
    n_cols = min(3, len(numerical_cols))
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(numerical_cols):
        if i < len(axes):
            df[col].hist(ax=axes[i], bins=30, edgecolor='black', alpha=0.7)
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
    
    # Hide empty subplots
    for i in range(len(numerical_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('histograms.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Histograms saved as 'histograms.png'")

# --- 4. Box Plots (Detect outliers) ---
if len(numerical_cols) > 0:
    fig, axes = plt.subplots(1, len(numerical_cols), figsize=(5*len(numerical_cols), 6))
    if len(numerical_cols) == 1:
        axes = [axes]
    
    for i, col in enumerate(numerical_cols):
        sns.boxplot(y=df[col], ax=axes[i])
        axes[i].set_title(f'Boxplot of {col}')
        
        # Add outlier statistics
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)][col]
        axes[i].text(0.02, 0.98, f'Outliers: {len(outliers)}', 
                    transform=axes[i].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('boxplots.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Box plots saved as 'boxplots.png'")

# --- 5. Correlation Heatmap ---
if len(numerical_cols) > 1:
    plt.figure(figsize=(12, 8))
    correlation_matrix = df[numerical_cols].corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='coolwarm', 
                fmt=".2f",
                center=0,
                square=True,
                mask=mask,
                cbar_kws={"shrink": .8})
    plt.title("Correlation Matrix (Numerical Variables)", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Correlation heatmap saved as 'correlation_heatmap.png'")

# --- 6. Categorical Analysis ---
if len(categorical_cols) > 0:
    n_cols = min(2, len(categorical_cols))
    n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(categorical_cols):
        if i < len(axes):
            # Limit to top 10 categories if too many
            value_counts = df[col].value_counts().head(10)
            
            sns.countplot(data=df[df[col].isin(value_counts.index)], 
                         y=col, ax=axes[i], order=value_counts.index)
            axes[i].set_title(f'Count plot of {col}')
            
            # Add count labels
            for container in axes[i].containers:
                axes[i].bar_label(container, label_type='edge')
    
    # Hide empty subplots
    for i in range(len(categorical_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('categorical_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Categorical plots saved as 'categorical_plots.png'")

# --- 7. Summary Report ---
print("\n" + "=" * 60)
print("📋 EDA SUMMARY REPORT")
print("=" * 60)
print(f"• Dataset contains {df.shape[0]:,} rows and {df.shape[1]} columns")
print(f"• {len(numerical_cols)} numerical columns: {list(numerical_cols)}")
print(f"• {len(categorical_cols)} categorical columns: {list(categorical_cols)}")
print(f"• Total missing values: {df.isnull().sum().sum()}")
print(f"• Duplicate rows: {df.duplicated().sum()}")

if len(numerical_cols) > 0:
    print(f"• Numerical data ranges:")
    for col in numerical_cols:
        print(f"  - {col}: {df[col].min():.2f} to {df[col].max():.2f}")

print("\n✅ EDA analysis complete! All plots have been saved.")
print("Files generated: histograms.png, boxplots.png, correlation_heatmap.png, categorical_plots.png")