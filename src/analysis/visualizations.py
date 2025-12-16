#!/usr/bin/env python3
"""Create beautiful visualizations for Filipino morphology LLM evaluation results."""

import json
import os
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Set style for clean, modern visualizations
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.labelcolor'] = '#333333'
plt.rcParams['text.color'] = '#333333'
plt.rcParams['xtick.color'] = '#666666'
plt.rcParams['ytick.color'] = '#666666'

# Color palette - warm coral/salmon tones
COLORS = {
    'primary': '#D97757',      # Coral/salmon
    'secondary': '#E8A87C',    # Light coral
    'tertiary': '#C38D9E',     # Dusty rose
    'accent': '#41B3A3',       # Teal accent
    'dark': '#3D405B',         # Dark slate
    'light': '#F7EDE2',        # Cream
    'gradient': ['#D97757', '#E8A87C', '#F6BD60', '#84A59D', '#41B3A3']
}

def load_all_results(results_dir: str) -> dict:
    """Load all evaluation results from model directories."""
    results = {}
    model_dirs = [d for d in glob(f"{results_dir}/*/") if "benchmark_evaluation" not in d]

    for model_dir in sorted(model_dirs):
        model_name = os.path.basename(model_dir.rstrip('/'))
        json_files = glob(f"{model_dir}/*.json")

        if json_files:
            latest = max(json_files, key=os.path.getmtime)
            try:
                with open(latest) as f:
                    data = json.load(f)
                results[model_name] = data
            except:
                pass

    return results


def create_dataframe(results: dict) -> pd.DataFrame:
    """Convert results to a pandas DataFrame for easier manipulation."""
    rows = []
    for model, data in results.items():
        benchmarks = data.get('benchmarks', {})
        row = {'model': model}
        for bench, scores in benchmarks.items():
            if 'contains_match_accuracy' in scores:
                row[bench] = scores['contains_match_accuracy'] * 100
            elif 'accuracy' in scores:
                row[bench] = scores['accuracy'] * 100
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.set_index('model')
    return df


def get_model_size(model_name: str) -> float:
    """Extract model size in billions from model name."""
    name_lower = model_name.lower()
    if '14b' in name_lower:
        return 14.0
    elif '9b' in name_lower:
        return 9.0
    elif '8b' in name_lower:
        return 8.0
    elif '7b' in name_lower:
        return 7.0
    elif '4b' in name_lower:
        return 4.0
    elif '3b' in name_lower:
        return 3.0
    elif '2b' in name_lower or '2.2b' in name_lower:
        return 2.0
    elif '1.5b' in name_lower:
        return 1.5
    elif '1b' in name_lower:
        return 1.0
    elif '0.5b' in name_lower:
        return 0.5
    elif 'large' in name_lower:
        return 0.8
    elif 'medium' in name_lower:
        return 0.35
    elif 'xl' in name_lower:
        return 1.5
    elif 'gpt2' in name_lower:
        return 0.12
    return 1.0


def get_model_family(model_name: str) -> str:
    """Extract model family from model name."""
    name_lower = model_name.lower()
    if 'sea-lion' in name_lower:
        return 'SEA-LION'
    elif 'qwen3' in name_lower:
        return 'Qwen3'
    elif 'qwen' in name_lower:
        return 'Qwen2.5'
    elif 'llama' in name_lower:
        return 'Llama3'
    elif 'gemma-2' in name_lower:
        return 'Gemma2'
    elif 'gemma' in name_lower:
        return 'Gemma1'
    elif 'gpt2' in name_lower:
        return 'GPT-2'
    return 'Other'


def plot_benchmark_comparison(df: pd.DataFrame, output_dir: str):
    """Create a horizontal bar chart comparing models across benchmarks."""
    benchmarks = ['pacute', 'cute', 'langgame', 'multi-digit-addition']
    benchmark_labels = {
        'pacute': 'PACUTE\n(Filipino Morphology)',
        'cute': 'CUTE\n(Character Understanding)',
        'langgame': 'LangGame\n(Subword Tasks)',
        'multi-digit-addition': 'Math\n(Multi-digit Addition)'
    }

    # Filter to benchmarks we have
    available = [b for b in benchmarks if b in df.columns]

    # Get top 15 models by average score
    df_subset = df[available].dropna(how='all')
    df_subset['avg'] = df_subset.mean(axis=1)
    df_subset = df_subset.nlargest(15, 'avg').drop('avg', axis=1)

    fig, ax = plt.subplots(figsize=(12, 10))

    y_positions = np.arange(len(df_subset))
    bar_height = 0.2

    for i, bench in enumerate(available):
        offset = (i - len(available)/2 + 0.5) * bar_height
        values = df_subset[bench].fillna(0)
        bars = ax.barh(y_positions + offset, values, bar_height,
                       label=benchmark_labels.get(bench, bench),
                       color=COLORS['gradient'][i % len(COLORS['gradient'])],
                       alpha=0.9, edgecolor='white', linewidth=0.5)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(df_subset.index, fontsize=10)
    ax.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Across Benchmarks', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim(0, 105)
    ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#cccccc')

    # Add gridlines
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/benchmark_comparison.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_dir}/benchmark_comparison.png")


def plot_heatmap(df: pd.DataFrame, output_dir: str):
    """Create a heatmap of all models vs all benchmarks."""
    benchmarks = ['pacute', 'cute', 'langgame', 'multi-digit-addition']
    available = [b for b in benchmarks if b in df.columns]

    df_heat = df[available].dropna(how='all')
    df_heat = df_heat.sort_values(by=available[0], ascending=True)

    # Rename columns for display
    col_labels = {
        'pacute': 'PACUTE',
        'cute': 'CUTE',
        'langgame': 'LangGame',
        'multi-digit-addition': 'Math'
    }
    df_heat = df_heat.rename(columns=col_labels)

    fig, ax = plt.subplots(figsize=(10, 14))

    # Custom colormap: cream to coral
    cmap = sns.light_palette(COLORS['primary'], as_cmap=True)

    sns.heatmap(df_heat, annot=True, fmt='.0f', cmap=cmap,
                linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'Accuracy (%)', 'shrink': 0.5},
                ax=ax, vmin=0, vmax=100)

    ax.set_title('Evaluation Results Heatmap', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Benchmark', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')

    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/heatmap.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_dir}/heatmap.png")


def plot_size_vs_performance(df: pd.DataFrame, output_dir: str):
    """Create scatter plot of model size vs performance."""
    df_plot = df.copy()
    df_plot['size'] = df_plot.index.map(get_model_size)
    df_plot['family'] = df_plot.index.map(get_model_family)
    df_plot['is_it'] = df_plot.index.str.contains('-it', case=False)

    if 'pacute' in df_plot.columns:
        metric = 'pacute'
        metric_label = 'PACUTE (Filipino Morphology)'
    else:
        metric = df_plot.columns[0]
        metric_label = metric

    df_plot = df_plot.dropna(subset=[metric])

    fig, ax = plt.subplots(figsize=(12, 8))

    families = df_plot['family'].unique()
    family_colors = {f: COLORS['gradient'][i % len(COLORS['gradient'])]
                     for i, f in enumerate(sorted(families))}

    for family in families:
        subset = df_plot[df_plot['family'] == family]

        # Base models (circles)
        base = subset[~subset['is_it']]
        if len(base) > 0:
            ax.scatter(base['size'], base[metric],
                      s=150, alpha=0.8, c=family_colors[family],
                      label=f'{family} (base)', marker='o',
                      edgecolors='white', linewidth=1.5)

        # Instruct models (diamonds)
        instruct = subset[subset['is_it']]
        if len(instruct) > 0:
            ax.scatter(instruct['size'], instruct[metric],
                      s=150, alpha=0.8, c=family_colors[family],
                      label=f'{family} (instruct)', marker='D',
                      edgecolors='white', linewidth=1.5)

    ax.set_xscale('log')
    ax.set_xlabel('Model Size (Billions of Parameters)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric_label} Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Size vs Filipino Morphology Performance', fontsize=16, fontweight='bold', pad=20)

    ax.set_xlim(0.08, 20)
    ax.set_ylim(15, 50)

    # Custom x-axis labels
    ax.set_xticks([0.1, 0.5, 1, 2, 5, 10, 15])
    ax.set_xticklabels(['0.1B', '0.5B', '1B', '2B', '5B', '10B', '15B'])

    ax.legend(loc='upper left', frameon=True, facecolor='white',
              edgecolor='#cccccc', ncol=2, fontsize=9)

    ax.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/size_vs_performance.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_dir}/size_vs_performance.png")


def plot_family_comparison(df: pd.DataFrame, output_dir: str):
    """Create grouped bar chart comparing model families."""
    df_plot = df.copy()
    df_plot['family'] = df_plot.index.map(get_model_family)

    benchmarks = ['pacute', 'cute', 'langgame', 'multi-digit-addition']
    available = [b for b in benchmarks if b in df_plot.columns]

    # Calculate mean by family
    family_means = df_plot.groupby('family')[available].mean()

    # Rename for display
    col_labels = {
        'pacute': 'PACUTE',
        'cute': 'CUTE',
        'langgame': 'LangGame',
        'multi-digit-addition': 'Math'
    }
    family_means = family_means.rename(columns=col_labels)

    # Sort by PACUTE performance
    if 'PACUTE' in family_means.columns:
        family_means = family_means.sort_values('PACUTE', ascending=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(family_means))
    width = 0.2

    for i, col in enumerate(family_means.columns):
        offset = (i - len(family_means.columns)/2 + 0.5) * width
        bars = ax.bar(x + offset, family_means[col], width,
                     label=col, color=COLORS['gradient'][i % len(COLORS['gradient'])],
                     alpha=0.9, edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(family_means.index, fontsize=11, fontweight='bold')
    ax.set_ylabel('Average Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Family Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#cccccc')

    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/family_comparison.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_dir}/family_comparison.png")


def plot_top_models(df: pd.DataFrame, output_dir: str):
    """Create a clean bar chart of top models on PACUTE benchmark."""
    if 'pacute' not in df.columns:
        return

    top_models = df.nlargest(10, 'pacute')[['pacute']].copy()
    top_models = top_models.sort_values('pacute', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = [COLORS['primary'] if '-it' in name else COLORS['secondary']
              for name in top_models.index]

    bars = ax.barh(range(len(top_models)), top_models['pacute'],
                   color=colors, alpha=0.9, edgecolor='white', linewidth=0.5)

    ax.set_yticks(range(len(top_models)))
    ax.set_yticklabels(top_models.index, fontsize=11)
    ax.set_xlabel('PACUTE Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Top 10 Models on Filipino Morphology (PACUTE)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim(0, 50)

    # Add value labels
    for i, (idx, row) in enumerate(top_models.iterrows()):
        ax.text(row['pacute'] + 0.5, i, f'{row["pacute"]:.1f}%',
                va='center', fontsize=10, color=COLORS['dark'])

    # Legend for instruct vs base
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['primary'], label='Instruct-tuned', alpha=0.9),
        mpatches.Patch(facecolor=COLORS['secondary'], label='Base model', alpha=0.9)
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True,
              facecolor='white', edgecolor='#cccccc')

    ax.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/top_models_pacute.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_dir}/top_models_pacute.png")


def create_all_visualizations(results_dir: str, output_dir: str, fmt: str = 'png'):
    """
    Create all visualizations from evaluation results.
    
    Args:
        results_dir: Directory containing evaluation results JSON files
        output_dir: Output directory for plots
        fmt: Output format ('png', 'pdf', 'svg')
    """
    # Update matplotlib format if needed
    if fmt != 'png':
        import matplotlib
        matplotlib.rcParams['savefig.format'] = fmt
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load results
    print("Loading evaluation results...")
    results = load_all_results(results_dir)
    if not results:
        print("❌ No results found!")
        return
    print(f"Found {len(results)} models with results")
    
    # Create dataframe
    df = create_dataframe(results)
    print(f"DataFrame shape: {df.shape}")
    print(f"Benchmarks: {list(df.columns)}")
    
    # Generate all visualizations
    print("\nGenerating visualizations...")
    plot_benchmark_comparison(df, output_dir)
    plot_heatmap(df, output_dir)
    plot_size_vs_performance(df, output_dir)
    plot_family_comparison(df, output_dir)
    plot_top_models(df, output_dir)
    
    print(f"\n✓ All visualizations saved to: {output_dir}")


def main():
    """CLI entry point for visualization creation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Create beautiful visualizations for Filipino morphology LLM evaluation results'
    )
    parser.add_argument('results_dir', type=str, 
                       help='Directory containing evaluation results JSON files')
    parser.add_argument('--output', type=str, default='plots', 
                       help='Output directory for plots')
    parser.add_argument('--format', type=str, choices=['png', 'pdf', 'svg'], default='png',
                       help='Output format for plots')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print(" Creating Visualizations for Filipino Morphology LLM Evaluation")
    print("="*80 + "\n")
    
    print(f"📁 Results directory: {args.results_dir}")
    print(f"📊 Output directory: {args.output}")
    print(f"🎨 Format: {args.format}\n")
    
    create_all_visualizations(args.results_dir, args.output, args.format)


if __name__ == "__main__":
    main()
