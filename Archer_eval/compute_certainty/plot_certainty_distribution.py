#!/usr/bin/env python

# Archer_eval/tools/plot_certainty_distribution.py

# -*- coding: utf-8 -*-
"""
Plot certainty metrics distribution vs correctness
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from scipy import stats
import re

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.facecolor'] = 'white'


def detect_dataset_info(input_path: str) -> dict:
    """
    Detect dataset name and length from input file path.
    Returns dict with 'dataset', 'length', 'prefix' keys.
    """
    input_path = str(input_path)
    filename = Path(input_path).name.lower()
    parent_dir = Path(input_path).parent.name.lower()
    
    # Detect dataset
    dataset = 'unknown'
    if 'aime2024' in filename or 'aime24' in filename:
        dataset = 'aime24'
    elif 'aime2025' in filename or 'aime25' in filename:
        dataset = 'aime25'
    elif 'livecodebench_v5' in filename or 'lcb_v5' in filename:
        dataset = 'lcb_v5'
    elif 'livecodebench_v6' in filename or 'lcb_v6' in filename:
        dataset = 'lcb_v6'
    elif 'minerva' in filename:
        dataset = 'minerva'
    
    # Detect length from parent directory (output vs output_16k)
    length = '8k'
    if '16k' in parent_dir or '_16k' in input_path:
        length = '16k'
    
    # Create prefix
    prefix = f'{dataset}_{length}'
    
    return {
        'dataset': dataset,
        'length': length,
        'prefix': prefix,
        'display_name': f'{dataset.upper()} ({length})'
    }


def plot_certainty_distribution(df, metric, output_dir, n_bins=10, prefix='', display_name=''):
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
    title = f'{metric.replace("_", " ").title()} Distribution'
    if display_name:
        title = f'{display_name}: {title}'
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = f'{prefix}_{metric}_distribution.png' if prefix else f'{metric}_distribution.png'
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


def plot_accuracy_by_bin(df, metric, output_dir, n_bins=10, prefix='', display_name=''):
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
    title = f'Accuracy by {metric.replace("_", " ").title()}'
    if display_name:
        title = f'{display_name}: {title}'
    ax.set_title(title, fontsize=14)
    ax.set_ylim(0, min(100, max(accuracies) * 1.3) if accuracies else 100)
    
    # Add horizontal line for overall accuracy
    overall_acc = df['is_correct'].mean() * 100
    ax.axhline(y=overall_acc, color='#e74c3c', linestyle='--', linewidth=2, 
               label=f'Overall Accuracy: {overall_acc:.1f}%')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = f'{prefix}_{metric}_accuracy.png' if prefix else f'{metric}_accuracy.png'
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


def plot_all_metrics_comparison(df, output_dir, prefix='', display_name=''):
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
    
    suptitle = 'Certainty Metrics Distribution: Correct vs Incorrect'
    if display_name:
        suptitle = f'{display_name}: {suptitle}'
    plt.suptitle(suptitle, fontsize=14, y=1.02)
    plt.tight_layout()
    
    filename = f'{prefix}_all_metrics_comparison.png' if prefix else 'all_metrics_comparison.png'
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


def plot_stacked_distribution(df, metric, output_dir, n_bins=15, prefix='', display_name=''):
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
    title = f'Stacked Distribution by {metric.replace("_", " ").title()}'
    if display_name:
        title = f'{display_name}: {title}'
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    filename = f'{prefix}_{metric}_stacked.png' if prefix else f'{metric}_stacked.png'
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


def plot_histogram_kde(df, metric, output_dir, n_bins=20, prefix='', display_name=''):
    """
    Plot histogram with KDE overlay for correct/incorrect samples
    Shows clear bar charts (percentage) with KDE curves overlaid
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    correct_df = df[df['is_correct'] == 1.0]
    incorrect_df = df[df['is_correct'] == 0.0]
    
    # Determine range
    all_values = df[metric].values
    vmin, vmax = all_values.min(), all_values.max()
    x_range = np.linspace(vmin, vmax, 200)
    
    # Create bins
    bins = np.linspace(vmin, vmax, n_bins + 1)
    bin_width = bins[1] - bins[0]
    
    # Calculate histograms as counts, then convert to percentage
    correct_hist_counts, _ = np.histogram(correct_df[metric], bins=bins)
    incorrect_hist_counts, _ = np.histogram(incorrect_df[metric], bins=bins)
    
    # Convert to percentage (within each group)
    correct_hist_pct = correct_hist_counts / len(correct_df) * 100 if len(correct_df) > 0 else correct_hist_counts
    incorrect_hist_pct = incorrect_hist_counts / len(incorrect_df) * 100 if len(incorrect_df) > 0 else incorrect_hist_counts
    
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    bar_width = bin_width * 0.4
    ax.bar(bin_centers - bar_width/2, correct_hist_pct, width=bar_width, 
           label='Correct', color='#2ecc71', alpha=0.75, edgecolor='white', linewidth=0.5)
    ax.bar(bin_centers + bar_width/2, incorrect_hist_pct, width=bar_width, 
           label='Incorrect', color='#e74c3c', alpha=0.75, edgecolor='white', linewidth=0.5)
    
    # Compute and plot KDE (scaled to percentage)
    # KDE density * bin_width * 100 = percentage scale
    scale_factor = bin_width * 100
    
    if len(correct_df) > 1:
        try:
            kde_correct = stats.gaussian_kde(correct_df[metric].values)
            ax.plot(x_range, kde_correct(x_range) * scale_factor, color='#1e8449', linewidth=2.5,
                   label=f'Correct KDE (μ={correct_df[metric].mean():.3f})')
        except Exception:
            pass
    
    if len(incorrect_df) > 1:
        try:
            kde_incorrect = stats.gaussian_kde(incorrect_df[metric].values)
            ax.plot(x_range, kde_incorrect(x_range) * scale_factor, color='#922b21', linewidth=2.5,
                   label=f'Incorrect KDE (μ={incorrect_df[metric].mean():.3f})')
        except Exception:
            pass
    
    # Add mean lines
    c_mean = correct_df[metric].mean()
    i_mean = incorrect_df[metric].mean()
    ax.axvline(c_mean, color='#1e8449', linestyle='--', linewidth=2, alpha=0.8)
    ax.axvline(i_mean, color='#922b21', linestyle='--', linewidth=2, alpha=0.8)
    
    ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=14)
    ax.set_ylabel('Percentage (%)', fontsize=14)
    # No title
    ax.legend(loc='upper right', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    filename = f'{prefix}_{metric}_hist_kde.pdf' if prefix else f'{metric}_hist_kde.pdf'
    output_path = output_dir / filename
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


def plot_all_metrics_kde(df, output_dir, prefix='', display_name=''):
    """
    Plot all metrics comparison with Histogram + KDE in one figure (2x2 grid)
    Shows clear bar charts with KDE curves overlaid
    All subplots share the same y-axis range for easy comparison
    """
    metrics = ['self_certainty', 'entropy', 'prob_disparity', 'trajectory_entropy']
    available_metrics = [m for m in metrics if m in df.columns]
    
    n_metrics = len(available_metrics)
    if n_metrics == 0:
        print("No metrics available for KDE comparison plot")
        return
    
    # Determine grid size
    ncols = 2
    nrows = (n_metrics + 1) // 2
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    correct_df = df[df['is_correct'] == 1.0]
    incorrect_df = df[df['is_correct'] == 0.0]
    
    n_bins = 15
    
    # First pass: compute all data and find global y_max
    plot_data = []
    global_y_max = 0
    
    for metric in available_metrics:
        vmin = df[metric].min()
        vmax = df[metric].max()
        x_range = np.linspace(vmin, vmax, 200)
        
        bins = np.linspace(vmin, vmax, n_bins + 1)
        bin_width = bins[1] - bins[0]
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        correct_hist_counts, _ = np.histogram(correct_df[metric], bins=bins)
        incorrect_hist_counts, _ = np.histogram(incorrect_df[metric], bins=bins)
        
        correct_hist_pct = correct_hist_counts / len(correct_df) * 100 if len(correct_df) > 0 else correct_hist_counts
        incorrect_hist_pct = incorrect_hist_counts / len(incorrect_df) * 100 if len(incorrect_df) > 0 else incorrect_hist_counts
        
        # Calculate KDE values
        scale_factor = bin_width * 100
        kde_correct_vals = None
        kde_incorrect_vals = None
        
        if len(correct_df) > 1:
            try:
                kde_correct = stats.gaussian_kde(correct_df[metric].values)
                kde_correct_vals = kde_correct(x_range) * scale_factor
            except Exception:
                pass
        
        if len(incorrect_df) > 1:
            try:
                kde_incorrect = stats.gaussian_kde(incorrect_df[metric].values)
                kde_incorrect_vals = kde_incorrect(x_range) * scale_factor
            except Exception:
                pass
        
        # Find max y value for this metric
        local_max = max(correct_hist_pct.max(), incorrect_hist_pct.max())
        if kde_correct_vals is not None:
            local_max = max(local_max, kde_correct_vals.max())
        if kde_incorrect_vals is not None:
            local_max = max(local_max, kde_incorrect_vals.max())
        
        global_y_max = max(global_y_max, local_max)
        
        plot_data.append({
            'metric': metric,
            'vmin': vmin, 'vmax': vmax,
            'x_range': x_range,
            'bins': bins, 'bin_width': bin_width, 'bin_centers': bin_centers,
            'correct_hist_pct': correct_hist_pct,
            'incorrect_hist_pct': incorrect_hist_pct,
            'kde_correct_vals': kde_correct_vals,
            'kde_incorrect_vals': kde_incorrect_vals,
            'c_mean': correct_df[metric].mean(),
            'i_mean': incorrect_df[metric].mean()
        })
    
    # Add some padding to y_max
    global_y_max = global_y_max * 1.1
    
    # Second pass: plot with unified y-axis
    for idx, data in enumerate(plot_data):
        ax = axes[idx]
        metric = data['metric']
        
        # Plot bar histograms (side by side)
        bar_width = data['bin_width'] * 0.4
        ax.bar(data['bin_centers'] - bar_width/2, data['correct_hist_pct'], width=bar_width, 
               label='Correct', color='#2ecc71', alpha=0.7, edgecolor='white', linewidth=0.5)
        ax.bar(data['bin_centers'] + bar_width/2, data['incorrect_hist_pct'], width=bar_width, 
               label='Incorrect', color='#e74c3c', alpha=0.7, edgecolor='white', linewidth=0.5)
        
        # Plot KDE
        if data['kde_correct_vals'] is not None:
            ax.plot(data['x_range'], data['kde_correct_vals'], color='#1e8449', linewidth=2.5,
                   label=f'Correct KDE')
        
        if data['kde_incorrect_vals'] is not None:
            ax.plot(data['x_range'], data['kde_incorrect_vals'], color='#922b21', linewidth=2.5,
                   label=f'Incorrect KDE')
        
        # Add mean lines
        ax.axvline(data['c_mean'], color='#1e8449', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.axvline(data['i_mean'], color='#922b21', linestyle='--', linewidth=1.5, alpha=0.8)
        
        # Set unified y-axis range
        ax.set_ylim(0, global_y_max)
        
        # Add text annotation for means (using global_y_max for consistent positioning)
        ax.text(data['c_mean'], global_y_max * 0.95, f'μ={data["c_mean"]:.3f}', color='#1e8449', 
                fontsize=11, ha='center', va='top', fontweight='bold')
        ax.text(data['i_mean'], global_y_max * 0.85, f'μ={data["i_mean"]:.3f}', color='#922b21', 
                fontsize=11, ha='center', va='top', fontweight='bold')
        
        ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel('Percentage (%)', fontsize=12)
        # No subplot title
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Hide extra axes if odd number of metrics
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    # No suptitle
    plt.tight_layout()
    
    filename = f'{prefix}_all_metrics_hist_kde.pdf' if prefix else 'all_metrics_hist_kde.pdf'
    output_path = output_dir / filename
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


def plot_kde_only(df, metric, output_dir, prefix='', display_name=''):
    """
    Plot KDE only (no histogram) for cleaner visualization
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    correct_df = df[df['is_correct'] == 1.0]
    incorrect_df = df[df['is_correct'] == 0.0]
    
    # Determine range
    vmin = df[metric].min()
    vmax = df[metric].max()
    x_range = np.linspace(vmin, vmax, 300)
    
    # Plot KDE with filled area
    if len(correct_df) > 1:
        try:
            kde_correct = stats.gaussian_kde(correct_df[metric].values)
            y_correct = kde_correct(x_range)
            ax.fill_between(x_range, y_correct, alpha=0.3, color='#2ecc71', label='Correct')
            ax.plot(x_range, y_correct, color='#27ae60', linewidth=2)
        except Exception:
            pass
    
    if len(incorrect_df) > 1:
        try:
            kde_incorrect = stats.gaussian_kde(incorrect_df[metric].values)
            y_incorrect = kde_incorrect(x_range)
            ax.fill_between(x_range, y_incorrect, alpha=0.3, color='#e74c3c', label='Incorrect')
            ax.plot(x_range, y_incorrect, color='#c0392b', linewidth=2)
        except Exception:
            pass
    
    # Add mean lines
    c_mean = correct_df[metric].mean()
    i_mean = incorrect_df[metric].mean()
    c_std = correct_df[metric].std()
    i_std = incorrect_df[metric].std()
    
    ax.axvline(c_mean, color='#27ae60', linestyle='--', linewidth=2, alpha=0.8,
               label=f'Correct μ={c_mean:.3f} (σ={c_std:.3f})')
    ax.axvline(i_mean, color='#c0392b', linestyle='--', linewidth=2, alpha=0.8,
               label=f'Incorrect μ={i_mean:.3f} (σ={i_std:.3f})')
    
    ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    title = f'{metric.replace("_", " ").title()} KDE Distribution'
    if display_name:
        title = f'{display_name}: {title}'
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = f'{prefix}_{metric}_kde.png' if prefix else f'{metric}_kde.png'
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


def plot_combined_kde_grid(df, output_dir, prefix='', display_name=''):
    """
    Plot all metrics KDE in a single figure with a cleaner style
    """
    metrics = ['self_certainty', 'entropy', 'prob_disparity', 'trajectory_entropy']
    available_metrics = [m for m in metrics if m in df.columns]
    
    n_metrics = len(available_metrics)
    if n_metrics == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    correct_df = df[df['is_correct'] == 1.0]
    incorrect_df = df[df['is_correct'] == 0.0]
    
    # Color palette
    colors = {
        'correct': {'fill': '#2ecc71', 'line': '#27ae60'},
        'incorrect': {'fill': '#e74c3c', 'line': '#c0392b'}
    }
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        
        vmin = df[metric].min()
        vmax = df[metric].max()
        x_range = np.linspace(vmin, vmax, 300)
        
        # KDE for correct
        if len(correct_df) > 1:
            try:
                kde_c = stats.gaussian_kde(correct_df[metric].values)
                y_c = kde_c(x_range)
                ax.fill_between(x_range, y_c, alpha=0.25, color=colors['correct']['fill'])
                ax.plot(x_range, y_c, color=colors['correct']['line'], linewidth=2.5, 
                       label='Correct')
            except Exception:
                pass
        
        # KDE for incorrect
        if len(incorrect_df) > 1:
            try:
                kde_i = stats.gaussian_kde(incorrect_df[metric].values)
                y_i = kde_i(x_range)
                ax.fill_between(x_range, y_i, alpha=0.25, color=colors['incorrect']['fill'])
                ax.plot(x_range, y_i, color=colors['incorrect']['line'], linewidth=2.5,
                       label='Incorrect')
            except Exception:
                pass
        
        # Statistics
        c_mean, c_std = correct_df[metric].mean(), correct_df[metric].std()
        i_mean, i_std = incorrect_df[metric].mean(), incorrect_df[metric].std()
        
        # Mean lines
        ax.axvline(c_mean, color=colors['correct']['line'], linestyle='--', 
                   linewidth=1.5, alpha=0.7)
        ax.axvline(i_mean, color=colors['incorrect']['line'], linestyle='--', 
                   linewidth=1.5, alpha=0.7)
        
        # Stats text box
        stats_text = f'Correct:   μ={c_mean:.3f}, σ={c_std:.3f}\nIncorrect: μ={i_mean:.3f}, σ={i_std:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Hide unused axes
    for idx in range(n_metrics, 4):
        axes[idx].set_visible(False)
    
    # Overall stats
    overall_acc = len(correct_df) / len(df) * 100
    title_prefix = f'{display_name}: ' if display_name else ''
    fig.suptitle(f'{title_prefix}Certainty Metrics KDE Distribution\n(Total: {len(df)}, Correct: {len(correct_df)}, Accuracy: {overall_acc:.1f}%)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    filename = f'{prefix}_all_metrics_kde_grid.png' if prefix else 'all_metrics_kde_grid.png'
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


# Metric display names mapping
METRIC_DISPLAY_NAMES = {
    'self_certainty': 'Self-Certainty',
    'entropy': 'Token Entropy',
    'prob_disparity': 'Probability Disparity',
    'trajectory_entropy': 'Trajectory Entropy'
}


def compute_mann_whitney(correct_values, incorrect_values):
    """
    Compute Mann-Whitney U test p-value and effect size r.
    
    Args:
        correct_values: array of metric values for correct samples
        incorrect_values: array of metric values for incorrect samples
    
    Returns:
        tuple: (p_value, effect_size_r)
    """
    stat, p_value = stats.mannwhitneyu(
        correct_values, 
        incorrect_values,
        alternative='two-sided'
    )
    
    n1, n2 = len(correct_values), len(incorrect_values)
    N = n1 + n2
    
    # Calculate Z value and effect size r
    mean_U = n1 * n2 / 2
    std_U = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    Z = (stat - mean_U) / std_U
    r = abs(Z) / np.sqrt(N)
    
    return p_value, r


def plot_all_metrics_1x4(df, output_dir, n_bins=12, prefix='', show_stats=True):
    """
    Plot 1x4 horizontal layout with shared y-axis and Mann-Whitney U test statistics.
    
    Features:
    - 1x4 horizontal layout (compact height: 3.5 inches)
    - Shared y-axis (only leftmost shows ylabel)
    - Custom metric names: Self-Certainty, Token Entropy, Probability Disparity, Trajectory Entropy
    - Enlarged legend (15pt)
    - Enlarged U test statistics (14pt)
    - No bold xlabel
    """
    metrics = ['self_certainty', 'entropy', 'prob_disparity', 'trajectory_entropy']
    available_metrics = [m for m in metrics if m in df.columns]
    
    if len(available_metrics) == 0:
        print("No metrics available for 1x4 plot")
        return
    
    correct_df = df[df['is_correct'] == 1.0]
    incorrect_df = df[df['is_correct'] == 0.0]
    
    # 1x4 layout with shared y-axis, compact height
    fig, axes = plt.subplots(1, 4, figsize=(24, 3.5), sharey=True)
    
    # Font sizes
    XLABEL_SIZE = 16
    YLABEL_SIZE = 16
    TICK_SIZE = 14
    LEGEND_SIZE = 15
    STATS_SIZE = 14
    
    # Colors
    COLOR_CORRECT = '#2ecc71'
    COLOR_INCORRECT = '#e74c3c'
    BAR_ALPHA = 0.8
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        
        all_values = df[metric].values
        vmin, vmax = all_values.min(), all_values.max()
        bins = np.linspace(vmin, vmax, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_width = bins[1] - bins[0]
        
        correct_hist, _ = np.histogram(correct_df[metric], bins=bins)
        incorrect_hist, _ = np.histogram(incorrect_df[metric], bins=bins)
        
        # Convert to percentage
        correct_pct = correct_hist / len(correct_df) * 100 if len(correct_df) > 0 else correct_hist
        incorrect_pct = incorrect_hist / len(incorrect_df) * 100 if len(incorrect_df) > 0 else incorrect_hist
        
        bar_width = bin_width * 0.4
        ax.bar(bin_centers - bar_width/2, correct_pct, width=bar_width, 
               label='Correct', color=COLOR_CORRECT, alpha=BAR_ALPHA, edgecolor='white')
        ax.bar(bin_centers + bar_width/2, incorrect_pct, width=bar_width, 
               label='Incorrect', color=COLOR_INCORRECT, alpha=BAR_ALPHA, edgecolor='white')
        
        # Mean lines
        c_mean = correct_df[metric].mean()
        i_mean = incorrect_df[metric].mean()
        ax.axvline(c_mean, color='#27ae60', linestyle='--', linewidth=2)
        ax.axvline(i_mean, color='#c0392b', linestyle='--', linewidth=2)
        
        # Custom metric name (no bold)
        display_name = METRIC_DISPLAY_NAMES.get(metric, metric.replace('_', ' ').title())
        ax.set_xlabel(display_name, fontsize=XLABEL_SIZE)
        
        # Only show ylabel on leftmost plot
        if idx == 0:
            ax.set_ylabel('Percentage (%)', fontsize=YLABEL_SIZE)
        
        ax.tick_params(axis='both', labelsize=TICK_SIZE)
        ax.legend(loc='upper right', fontsize=LEGEND_SIZE)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add Mann-Whitney U test statistics
        if show_stats and len(correct_df) > 0 and len(incorrect_df) > 0:
            p_value, r = compute_mann_whitney(correct_df[metric].values, incorrect_df[metric].values)
            stats_text = f'p = {p_value:.2e}\nr = {r:.3f}'
            props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=STATS_SIZE,
                    verticalalignment='top', bbox=props, fontfamily='monospace')
    
    # Hide unused axes if less than 4 metrics
    for idx in range(len(available_metrics), 4):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    filename = f'{prefix}_all_metrics_1x4.png' if prefix else 'all_metrics_1x4.png'
    output_path = output_dir / filename
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
    parser.add_argument('--kde-only', action='store_true',
                       help='Only plot KDE (skip histogram and other plots)')
    parser.add_argument('--prefix', '-p', type=str, default=None,
                       help='Prefix for output filenames (auto-detected if not specified)')
    parser.add_argument('--no-prefix', action='store_true',
                       help='Do not add prefix to output filenames')
    args = parser.parse_args()
    
    # Load data
    print(f'Loading data from: {args.input}')
    df = pd.read_parquet(args.input)
    print(f'Loaded {len(df)} samples')
    
    # Detect dataset info
    dataset_info = detect_dataset_info(args.input)
    
    # Determine prefix and display name
    if args.no_prefix:
        prefix = ''
        display_name = ''
    elif args.prefix:
        prefix = args.prefix
        display_name = args.prefix.upper().replace('_', ' ')
    else:
        prefix = dataset_info['prefix']
        display_name = dataset_info['display_name']
    
    print(f'Dataset: {dataset_info["dataset"]}, Length: {dataset_info["length"]}')
    if prefix:
        print(f'File prefix: {prefix}')
    
    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(args.input).parent / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output directory: {output_dir}')
    
    # Metrics to plot
    metrics = ['self_certainty', 'entropy', 'prob_disparity', 'trajectory_entropy']
    
    if args.kde_only:
        # Plot Histogram + KDE (with bar charts)
        print('\nPlotting Histogram + KDE distributions...')
        for metric in metrics:
            if metric in df.columns:
                print(f'  Plotting {metric} Histogram + KDE...')
                plot_histogram_kde(df, metric, output_dir, n_bins=args.bins,
                                  prefix=prefix, display_name=display_name)
        
        print('\nPlotting all metrics Histogram + KDE grid...')
        plot_all_metrics_kde(df, output_dir, prefix=prefix, display_name=display_name)
    else:
        # Plot each metric (all types)
        for metric in metrics:
            if metric in df.columns:
                print(f'\nPlotting {metric}...')
                plot_certainty_distribution(df, metric, output_dir, n_bins=args.bins, 
                                           prefix=prefix, display_name=display_name)
                plot_accuracy_by_bin(df, metric, output_dir, n_bins=args.bins,
                                    prefix=prefix, display_name=display_name)
                plot_stacked_distribution(df, metric, output_dir, n_bins=args.bins,
                                         prefix=prefix, display_name=display_name)
                plot_histogram_kde(df, metric, output_dir, n_bins=args.bins,
                                  prefix=prefix, display_name=display_name)
        
        # Plot all metrics comparison
        print('\nPlotting all metrics comparison (histogram)...')
        plot_all_metrics_comparison(df, output_dir, prefix=prefix, display_name=display_name)
        
        print('\nPlotting all metrics Histogram + KDE...')
        plot_all_metrics_kde(df, output_dir, prefix=prefix, display_name=display_name)
    
    # Always plot 1x4 layout with Mann-Whitney U test
    print('\nPlotting 1x4 layout with Mann-Whitney U test...')
    plot_all_metrics_1x4(df, output_dir, n_bins=args.bins, prefix=prefix, show_stats=True)
    
    # Print summary
    print('\n' + '='*60)
    print(f'Summary Statistics - {display_name if display_name else "All Data"}')
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
            c_std = correct_df[metric].std()
            i_std = incorrect_df[metric].std()
            diff = c_mean - i_mean
            print(f'  {metric:20s}: {c_mean:.4f} (σ={c_std:.4f}) vs {i_mean:.4f} (σ={i_std:.4f}) | Δ={diff:+.4f}')
    
    print(f'\nPlots saved to: {output_dir}')


if __name__ == '__main__':
    main()
