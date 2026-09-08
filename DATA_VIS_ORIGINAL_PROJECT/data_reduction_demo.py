"""
Data Reduction Demonstration
Comparing Hourly vs Daily vs Monthly Aggregations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("colorblind")

print("="*80)
print("DATA REDUCTION DEMONSTRATION")
print("="*80)

# Create output directory
os.makedirs('figures/reduction', exist_ok=True)

# Load all three datasets
print("\n[1/3] Loading datasets...")
df_hourly = pd.read_csv('data_cleaned_hourly.csv')
df_daily = pd.read_csv('data_cleaned_daily.csv')
df_monthly = pd.read_csv('data_cleaned_monthly.csv')

# Parse dates
df_hourly['Formatted Date'] = pd.to_datetime(df_hourly['Formatted Date'])
df_daily['Date'] = pd.to_datetime(df_daily['Date'])
df_monthly['Date'] = pd.to_datetime(df_monthly['Date'])

print(f"✓ Hourly dataset: {len(df_hourly):,} observations")
print(f"✓ Daily dataset: {len(df_daily):,} observations")
print(f"✓ Monthly dataset: {len(df_monthly):,} observations")

# Calculate reduction statistics
print("\n" + "="*80)
print("REDUCTION STATISTICS")
print("="*80)

reduction_stats = {
    'Dataset': ['Hourly (Original)', 'Daily', 'Monthly'],
    'Observations': [len(df_hourly), len(df_daily), len(df_monthly)],
    'Reduction %': [0, 
                    (1 - len(df_daily)/len(df_hourly))*100,
                    (1 - len(df_monthly)/len(df_hourly))*100],
    'File Size (MB)': [
        os.path.getsize('data_cleaned_hourly.csv') / (1024*1024),
        os.path.getsize('data_cleaned_daily.csv') / (1024*1024),
        os.path.getsize('data_cleaned_monthly.csv') / (1024*1024)
    ]
}

reduction_df = pd.DataFrame(reduction_stats)
print(reduction_df.to_string(index=False))

# Visualization 1: Side-by-side comparison of temperature trends
print("\n[2/3] Creating comparison visualizations...")
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

# Hourly data (sample 1 month for visibility)
sample_start = '2010-01-01'
sample_end = '2010-02-01'
hourly_sample = df_hourly[(df_hourly['Formatted Date'] >= sample_start) & 
                          (df_hourly['Formatted Date'] < sample_end)]
axes[0].plot(hourly_sample['Formatted Date'], hourly_sample['Temperature (C)'], 
             linewidth=0.5, alpha=0.7, color='#3498db')
axes[0].set_ylabel('Temperature (°C)', fontsize=11)
axes[0].set_title('Hourly Data (January 2010 Sample) - High Noise, Detailed Fluctuations', 
                  fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].text(0.02, 0.95, f'n = {len(hourly_sample):,} observations', 
             transform=axes[0].transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Daily data (same period)
daily_sample = df_daily[(df_daily['Date'] >= sample_start) & 
                        (df_daily['Date'] < sample_end)]
axes[1].plot(daily_sample['Date'], daily_sample['Temp_Mean'], 
             linewidth=1.5, color='#e74c3c', marker='o', markersize=3)
axes[1].set_ylabel('Temperature (°C)', fontsize=11)
axes[1].set_title('Daily Aggregated Data (January 2010) - Reduced Noise, Clear Daily Pattern', 
                  fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].text(0.02, 0.95, f'n = {len(daily_sample)} observations', 
             transform=axes[1].transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Monthly data (full dataset)
axes[2].plot(df_monthly['Date'], df_monthly['Temp_Mean'], 
             linewidth=2, color='#2ecc71', marker='s', markersize=5)
axes[2].fill_between(df_monthly['Date'],
                     df_monthly['Temp_Mean'] - df_monthly['Temp_Std'],
                     df_monthly['Temp_Mean'] + df_monthly['Temp_Std'],
                     alpha=0.3, color='#2ecc71')
axes[2].set_ylabel('Temperature (°C)', fontsize=11)
axes[2].set_xlabel('Date', fontsize=11)
axes[2].set_title('Monthly Aggregated Data (Full Dataset) - Smooth Trends, Seasonal Patterns', 
                  fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3)
axes[2].text(0.02, 0.95, f'n = {len(df_monthly)} observations', 
             transform=axes[2].transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('figures/reduction/comparison_temperature_trends.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/reduction/comparison_temperature_trends.png")

# Visualization 2: Noise reduction demonstration
print("\n[3/3] Creating noise reduction analysis...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Calculate rolling statistics for hourly data
hourly_2010 = df_hourly[df_hourly['Formatted Date'].dt.year == 2010].copy()
hourly_2010['Day'] = hourly_2010['Formatted Date'].dt.dayofyear
daily_2010 = df_daily[df_daily['Date'].dt.year == 2010].copy()
daily_2010['Day'] = daily_2010['Date'].dt.dayofyear

# Plot 1: Variance comparison
variance_stats = {
    'Aggregation Level': ['Hourly', 'Daily', 'Monthly'],
    'Temperature Variance': [
        df_hourly['Temperature (C)'].var(),
        df_daily['Temp_Mean'].var(),
        df_monthly['Temp_Mean'].var()
    ]
}
axes[0, 0].bar(variance_stats['Aggregation Level'], variance_stats['Temperature Variance'],
               color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.7)
axes[0, 0].set_ylabel('Variance (°C²)', fontsize=11)
axes[0, 0].set_title('Temperature Variance by Aggregation Level', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Plot 2: Data points comparison
axes[0, 1].bar(reduction_df['Dataset'], reduction_df['Observations'],
               color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.7)
axes[0, 1].set_ylabel('Number of Observations', fontsize=11)
axes[0, 1].set_title('Dataset Size Reduction', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(reduction_df['Observations']):
    axes[0, 1].text(i, v + 1000, f'{v:,}', ha='center', fontsize=9)

# Plot 3: File size comparison
axes[1, 0].bar(reduction_df['Dataset'], reduction_df['File Size (MB)'],
               color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.7)
axes[1, 0].set_ylabel('File Size (MB)', fontsize=11)
axes[1, 0].set_title('Storage Space Reduction', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(reduction_df['File Size (MB)']):
    axes[1, 0].text(i, v + 0.05, f'{v:.2f} MB', ha='center', fontsize=9)

# Plot 4: Statistical properties preservation
mean_comparison = {
    'Dataset': ['Hourly', 'Daily', 'Monthly'],
    'Mean Temp': [
        df_hourly['Temperature (C)'].mean(),
        df_daily['Temp_Mean'].mean(),
        df_monthly['Temp_Mean'].mean()
    ]
}
axes[1, 1].bar(mean_comparison['Dataset'], mean_comparison['Mean Temp'],
               color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.7)
axes[1, 1].set_ylabel('Mean Temperature (°C)', fontsize=11)
axes[1, 1].set_title('Mean Temperature Preservation', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')
axes[1, 1].set_ylim([min(mean_comparison['Mean Temp'])-1, max(mean_comparison['Mean Temp'])+1])
for i, v in enumerate(mean_comparison['Mean Temp']):
    axes[1, 1].text(i, v + 0.1, f'{v:.2f}°C', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('figures/reduction/noise_reduction_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/reduction/noise_reduction_analysis.png")

# Generate detailed statistics report
print("\n" + "="*80)
print("DETAILED REDUCTION ANALYSIS")
print("="*80)

print("\n📊 STATISTICAL PROPERTIES PRESERVATION:")
print(f"  Hourly Mean Temperature: {df_hourly['Temperature (C)'].mean():.4f}°C")
print(f"  Daily Mean Temperature:  {df_daily['Temp_Mean'].mean():.4f}°C")
print(f"  Monthly Mean Temperature: {df_monthly['Temp_Mean'].mean():.4f}°C")
print(f"  → Difference (Hourly vs Monthly): {abs(df_hourly['Temperature (C)'].mean() - df_monthly['Temp_Mean'].mean()):.4f}°C")

print("\n📊 NOISE REDUCTION:")
print(f"  Hourly Std Dev: {df_hourly['Temperature (C)'].std():.4f}°C")
print(f"  Daily Std Dev:  {df_daily['Temp_Mean'].std():.4f}°C")
print(f"  Monthly Std Dev: {df_monthly['Temp_Mean'].std():.4f}°C")
print(f"  → Noise reduction: {(1 - df_monthly['Temp_Mean'].std()/df_hourly['Temperature (C)'].std())*100:.1f}%")

print("\n📊 EFFICIENCY GAINS:")
print(f"  Storage reduction (Monthly vs Hourly): {reduction_df.iloc[2]['Reduction %']:.1f}%")
print(f"  Processing speed improvement: ~{len(df_hourly)/len(df_monthly):.0f}x faster")
print(f"  Visualization clarity: Significantly improved for trend analysis")

print("\n📊 BENEFITS OF AGGREGATION:")
print("  ✓ Reduced noise while preserving statistical properties")
print("  ✓ Clearer visualization of long-term trends")
print("  ✓ Faster processing and analysis")
print("  ✓ Smaller file sizes for sharing and storage")
print("  ✓ Better suited for interactive dashboards")
print("  ✓ Seasonal patterns more visible")

print("\n" + "="*80)
print("DATA REDUCTION DEMONSTRATION COMPLETE")
print("="*80)
