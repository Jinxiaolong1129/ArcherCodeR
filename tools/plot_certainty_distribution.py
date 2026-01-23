#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot certainty metrics distribution vs correctness
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14


def plot_certainty_distribution(df, metric, output_dir, n_bins=10):
    """
    Plot bar chart: x-axis = metric value, y-axis = count of correct/incorrect
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    correct_df = df[df['is_correct'] == 1.0]
    incorrect_df = df[df['is_correct'] == 0.0]
    
    # Determine bin range
    all_values = df[metric].values
    vmin, vmax = all_values.min(), all_values.max()
    
    # Create bins with nice round numbers
    bins = np.linspace(vmin, vmax, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    
    # Compute histograms
    correct_hist, _ = np.histogram(correct_df[metric], bins=bins)
    incorrect_hist, _ = np.histogram(incorrect_df[metric], bins=bins)
    
    # Plot
    bar_width = bin_width * 0.4
    
    ax.bar(bin_centers - bar_width/2, correct_hist, width=bar_width, 
           label='Correct', color='#2ecc71', alpha=0.85, edgecolor='white')
    ax.bar(bin_centers + bar_width/2, incorrect_hist, width=bar_width, 
           label='Incorrect', color='#e74c3c', alpha=0.85, edgecolor='white')
    
    # Set x-axis with clean tick marks
    ax.set_xlim(vmin - bin_width, vmax + bin_width)
    
    # Create nice round tick values
    tick_interval = (vmax - vmin) / 8
    # Round to nice numbers
    magnitude = 10 ** np.floor(np.log10(tick_interval))
    tick_interval = np.ceil(tick_interval / magnitude) * magnitude
    
    tick_start = np.floor(vmin / tick_interval) * tick_interval
    tick_end = np.ceil(vmax / tick_interval) * tick_interval
    ticks = np.arange(tick_start, tick_end + tick_interval, tick_interval)
    
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{t:.1f}' if t < 10 else f'{int(t)}' for t in ticks])
    
    ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Distribution of Correct/Incorrect Answers by {metric.replace("_", " ").title()}', fontsize=14)
    ax.legend(loc='upper right')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / f'{metric}_distribution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


def plot_accuracy_by_bin(df, metric, output_dir, n_bins=10):
    """
    Plot accuracy rate per bin with clean x-axis
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Determine bin range
    all_values = df[metric].values
    vmin, vmax = all_values.min(), all_values.max()
    
    # Create bins
    bins = np.linspace(vmin, vmax, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    
    # Calculate accuracy per bin
    accuracies = []
    counts = []
    for i in range(len(bins) - 1):
        mask = (df[metric] >= bins[i]) & (df[metric] < bins[i+1])
        if i == len(bins) - 2:  # Last bin includes right edge
            mask = (df[metric] >= bins[i]) & (df[metric] <= bins[i+1])
        bin_df = df[mask]
        if len(bin_df) > 0:
            accuracies.append(bin_df['is_correct'].mean() * 100)
            counts.append(len(bin_df))
        else:
            accuracies.append(0)
            counts.append(0)
    
    # Plot
    bars = ax.bar(bin_centers, accuracies, width=bin_width * 0.8, 
                  color='#3498db', alpha=0.85, edgecolor='white')
    
    # Add count annotations
    for bar, count, acc in zip(bars, counts, accuracies):
        if count > 0:
            ax.annotate(f'{acc:.1f}%\n(n={count})',
                       xy=(bar.get_x() + bar.get_width() / 2, acc),
                       xytext=(0, 5), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    # Set x-axis with clean tick marks
    ax.set_xlim(vmin - bin_width, vmax + bin_width)
    
    tick_interval = (vmax - vmin) / 8
    magnitude = 10 ** np.floor(np.log10(tick_interval))
    tick_interval = np.ceil(tick_interval / magnitude) * magnitude
    
    tick_start = np.floor(vmin / tick_interval) * tick_interval
    tick_end = np.ceil(vmax / tick_interval) * tick_interval
    ticks = np.arange(tick_start, tick_end + tick_interval, tick_interval)
    
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{t:.1f}' if t < 10 else f'{int(t)}' for t in ticks])
    
    ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Accuracy by {metric.replace("_", " ").title()}', fontsize=14)
    ax.set_ylim(0, min(100, max(accuracies) * 1.3) if accuracies else 100)
    
    # Add horizontal line for overall accuracy
    overall_acc = df['is_correct'].mean() * 100
    ax.axhline(y=overall_acc, color='#e74c3c', linestyle='--', linewidth=2, 
               label=f'Overall Accuracy: {overall_acc:.1f}%')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / f'{metric}_accuracy.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


def plot_all_metrics_comparison(df, output_dir):
    """
    Plot all metrics comparison in one figure
    """
    metrics = ['self_certainty', 'entropy', 'prob_disparity', 'trajectory_entropy']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        correct_df = df[df['is_correct'] == 1.0]
        incorrect_df = df[df['is_correct'] == 0.0]
        
        # Plot histograms
        ax.hist(correct_df[metric], bins=30, alpha=0.6, label='Correct', 
                color='#2ecc71', density=True, edgecolor='white')
        ax.hist(incorrect_df[metric], bins=30, alpha=0.6, label='Incorrect', 
                color='#e74c3c', density=True, edgecolor='white')
        
        # Add mean lines
        c_mean = correct_df[metric].mean()
        i_mean = incorrect_df[metric].mean()
        ax.axvline(c_mean, color='#27ae60', linestyle='--', linewidth=2, 
                   label=f'Correct Mean: {c_mean:.3f}')
        ax.axvline(i_mean, color='#c0392b', linestyle='--', linewidth=2,
                   label=f'Incorrect Mean: {i_mean:.3f}')
        
        ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{metric.replace("_", " ").title()} Distribution', fontsize=12)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Certainty Metrics Distribution: Correct vs Incorrect', fontsize=14, y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / 'all_metrics_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


def plot_stacked_distribution(df, metric, output_dir, n_bins=15):
    """
    Plot stacked bar chart showing proportion of correct/incorrect per bin
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Determine bin range
    all_values = df[metric].values
    vmin, vmax = all_values.min(), all_values.max()
    
    # Create bins
    bins = np.linspace(vmin, vmax, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    
    # Compute histograms
    correct_df = df[df['is_correct'] == 1.0]
    incorrect_df = df[df['is_correct'] == 0.0]
    
    correct_hist, _ = np.histogram(correct_df[metric], bins=bins)
    incorrect_hist, _ = np.histogram(incorrect_df[metric], bins=bins)
    
    # Plot stacked
    ax.bar(bin_centers, correct_hist, width=bin_width * 0.85, 
           label='Correct', color='#2ecc71', alpha=0.85, edgecolor='white')
    ax.bar(bin_centers, incorrect_hist, width=bin_width * 0.85, bottom=correct_hist,
           label='Incorrect', color='#e74c3c', alpha=0.85, edgecolor='white')
    
    # Set x-axis with clean tick marks
    ax.set_xlim(vmin - bin_width, vmax + bin_width)
    
    tick_interval = (vmax - vmin) / 8
    magnitude = 10 ** np.floor(np.log10(tick_interval))
    tick_interval = np.ceil(tick_interval / magnitude) * magnitude
    
    tick_start = np.floor(vmin / tick_interval) * tick_interval
    tick_end = np.ceil(vmax / tick_interval) * tick_interval
    ticks = np.arange(tick_start, tick_end + tick_interval, tick_interval)
    
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{t:.1f}' if t < 10 else f'{int(t)}' for t in ticks])
    
    ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Stacked Distribution by {metric.replace("_", " ").title()}', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = output_dir / f'{metric}_stacked.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Plot certainty distribution')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Path to certainty parquet file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output directory for plots (default: same as input)')
    parser.add_argument('--bins', '-b', type=int, default=15,
                       help='Number of bins (default: 15)')
    args = parser.parse_args()
    
    # Load data
    print(f'Loading data from: {args.input}')
    df = pd.read_parquet(args.input)
    print(f'Loaded {len(df)} samples')
    
    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(args.input).parent / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output directory: {output_dir}')
    
    # Metrics to plot
    metrics = ['self_certainty', 'entropy', 'prob_disparity', 'trajectory_entropy']
    
    # Plot each metric
    for metric in metrics:
        if metric in df.columns:
            print(f'\nPlotting {metric}...')
            plot_certainty_distribution(df, metric, output_dir, n_bins=args.bins)
            plot_accuracy_by_bin(df, metric, output_dir, n_bins=args.bins)
            plot_stacked_distribution(df, metric, output_dir, n_bins=args.bins)
    
    # Plot all metrics comparison
    print('\nPlotting all metrics comparison...')
    plot_all_metrics_comparison(df, output_dir)
    
    # Print summary
    print('\n' + '='*60)
    print('Summary Statistics')
    print('='*60)
    correct_df = df[df['is_correct'] == 1.0]
    incorrect_df = df[df['is_correct'] == 0.0]
    
    print(f'\nTotal: {len(df)}, Correct: {len(correct_df)}, Incorrect: {len(incorrect_df)}')
    print(f'Accuracy: {len(correct_df)/len(df)*100:.2f}%')
    
    print('\nMetric Comparison (Correct vs Incorrect):')
    for metric in metrics:
        if metric in df.columns:
            c_mean = correct_df[metric].mean()
            i_mean = incorrect_df[metric].mean()
            diff = c_mean - i_mean
            print(f'  {metric:20s}: {c_mean:.4f} vs {i_mean:.4f} (diff: {diff:+.4f})')
    
    print(f'\nPlots saved to: {output_dir}')


if __name__ == '__main__':
    main()
