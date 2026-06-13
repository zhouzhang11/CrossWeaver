#!/usr/bin/env python3
"""
Collect and analyze shift experiment results.
This script parses validation results from all shift experiments and generates:
1. CSV file with all results
2. Heatmaps showing mAP vs shift
3. Statistical analysis
"""

import os
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configuration
PROJECT_DIR = Path('DroneVehicle/shift_experiments')
OUTPUT_DIR = Path('DroneVehicle/shift_analysis')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_results_from_directory(result_dir):
    """Parse results from a validation output directory."""
    results_file = result_dir / 'results.csv'
    
    if not results_file.exists():
        return None
    
    try:
        # Read the last line of results.csv which contains the final metrics
        df = pd.read_csv(results_file)
        if len(df) == 0:
            return None
        
        # Get the last row (final epoch results)
        last_row = df.iloc[-1]
        
        # Extract metrics (column names may vary, adjust as needed)
        metrics = {}
        for col in df.columns:
            if 'mAP50' in col or 'mAP' in col or 'precision' in col or 'recall' in col:
                metrics[col] = last_row[col]
        
        return metrics
    except Exception as e:
        print(f"Error parsing {results_file}: {e}")
        return None

def collect_all_results():
    """Collect results from all shift experiment directories."""
    results = []
    
    if not PROJECT_DIR.exists():
        print(f"Error: Project directory {PROJECT_DIR} does not exist!")
        return results
    
    # Find all result directories
    pattern = 'shift_experiment_x*_y*'
    result_dirs = sorted(PROJECT_DIR.glob(pattern))
    
    print(f"Found {len(result_dirs)} result directories")
    
    for result_dir in result_dirs:
        # Parse shift values from directory name
        # Format: shift_experiment_x+5_y-10 or shift_experiment_x-5_y+10
        dir_name = result_dir.name
        try:
            parts = dir_name.replace('shift_experiment_', '').split('_')
            shift_x = int(parts[0].replace('x', ''))
            shift_y = int(parts[1].replace('y', ''))
            
            # Parse metrics from results
            metrics = parse_results_from_directory(result_dir)
            
            if metrics:
                result = {
                    'shift_x': shift_x,
                    'shift_y': shift_y,
                    'directory': str(result_dir),
                }
                result.update(metrics)
                results.append(result)
                print(f"✓ Collected: x={shift_x:+3d}, y={shift_y:+3d}")
            else:
                print(f"✗ No results: x={shift_x:+3d}, y={shift_y:+3d}")
        
        except Exception as e:
            print(f"Error processing {dir_name}: {e}")
            continue
    
    return results

def save_results_to_csv(results, output_file):
    """Save results to CSV file."""
    if not results:
        print("No results to save!")
        return
    
    df = pd.DataFrame(results)
    df = df.sort_values(['shift_x', 'shift_y'])
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    print(f"Total experiments: {len(df)}")
    return df

def create_heatmaps(df, output_dir):
    """Create heatmaps for different metrics."""
    # Find metric columns (excluding shift_x, shift_y, directory)
    metric_cols = [col for col in df.columns if col not in ['shift_x', 'shift_y', 'directory']]
    
    if not metric_cols:
        print("No metric columns found for heatmap!")
        return
    
    # Create a heatmap for each metric
    for metric in metric_cols:
        try:
            # Create pivot table
            pivot = df.pivot(index='shift_y', columns='shift_x', values=metric)
            
            # Create figure
            plt.figure(figsize=(14, 10))
            
            # Create heatmap
            sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn', 
                       center=pivot.mean().mean(), 
                       cbar_kws={'label': metric})
            
            plt.title(f'{metric} vs Spatial Shift\n(Positive = Right/Down, Negative = Left/Up)', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Shift X (pixels)', fontsize=12)
            plt.ylabel('Shift Y (pixels)', fontsize=12)
            plt.tight_layout()
            
            # Save figure
            output_file = output_dir / f'heatmap_{metric.replace("/", "_")}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved heatmap: {output_file}")
        
        except Exception as e:
            print(f"Error creating heatmap for {metric}: {e}")

def generate_statistics(df, output_file):
    """Generate statistical analysis of results."""
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SHIFT EXPERIMENT STATISTICAL ANALYSIS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total experiments: {len(df)}\n\n")
        
        # Find metric columns
        metric_cols = [col for col in df.columns if col not in ['shift_x', 'shift_y', 'directory']]
        
        for metric in metric_cols:
            f.write("-"*80 + "\n")
            f.write(f"Metric: {metric}\n")
            f.write("-"*80 + "\n")
            
            values = df[metric].values
            
            # Basic statistics
            f.write(f"Mean:     {values.mean():.6f}\n")
            f.write(f"Std Dev:  {values.std():.6f}\n")
            f.write(f"Min:      {values.min():.6f}\n")
            f.write(f"Max:      {values.max():.6f}\n")
            f.write(f"Range:    {values.max() - values.min():.6f}\n\n")
            
            # Best and worst cases
            best_idx = df[metric].idxmax()
            worst_idx = df[metric].idxmin()
            
            f.write(f"Best performance:\n")
            f.write(f"  Value: {df.loc[best_idx, metric]:.6f}\n")
            f.write(f"  Shift: x={df.loc[best_idx, 'shift_x']:+d}, y={df.loc[best_idx, 'shift_y']:+d}\n\n")
            
            f.write(f"Worst performance:\n")
            f.write(f"  Value: {df.loc[worst_idx, metric]:.6f}\n")
            f.write(f"  Shift: x={df.loc[worst_idx, 'shift_x']:+d}, y={df.loc[worst_idx, 'shift_y']:+d}\n\n")
            
            # Performance degradation from baseline (0, 0) if available
            baseline = df[(df['shift_x'] == 0) & (df['shift_y'] == 0)]
            if len(baseline) > 0:
                baseline_value = baseline[metric].values[0]
                degradation = ((values.mean() - baseline_value) / baseline_value) * 100
                f.write(f"Baseline (0, 0): {baseline_value:.6f}\n")
                f.write(f"Average degradation: {degradation:+.2f}%\n\n")
        
        f.write("="*80 + "\n")
    
    print(f"\nStatistics saved to: {output_file}")

def main():
    print("="*80)
    print("COLLECTING SHIFT EXPERIMENT RESULTS")
    print("="*80)
    print()
    
    # Collect results
    results = collect_all_results()
    
    if not results:
        print("\nNo results found! Make sure experiments have been run.")
        return
    
    # Save to CSV
    csv_file = OUTPUT_DIR / 'shift_results.csv'
    df = save_results_to_csv(results, csv_file)
    
    if df is None or len(df) == 0:
        print("No data to analyze!")
        return
    
    # Create heatmaps
    print("\nGenerating heatmaps...")
    create_heatmaps(df, OUTPUT_DIR)
    
    # Generate statistics
    print("\nGenerating statistics...")
    stats_file = OUTPUT_DIR / 'statistics.txt'
    generate_statistics(df, stats_file)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"  - Results CSV: {csv_file}")
    print(f"  - Heatmaps: heatmap_*.png")
    print(f"  - Statistics: {stats_file}")
    print("="*80)

if __name__ == '__main__':
    main()
