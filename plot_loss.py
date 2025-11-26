#!/usr/bin/env python3
"""
Plot training loss from log files.

Usage:
    python plot_loss.py logs/your_run_id.txt
    python plot_loss.py logs/your_run_id.txt --output loss_plot.png
"""

import re
import sys
import argparse
import matplotlib.pyplot as plt
from pathlib import Path


def parse_log_file(log_file):
    """Parse log file and extract step and validation loss values."""
    steps = []
    val_losses = []
    
    # Pattern to match: step:123/1000 val_loss:2.3456 ...
    pattern = re.compile(r'step:(\d+)/(\d+)\s+val_loss:([\d.]+)')
    
    with open(log_file, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                step = int(match.group(1))
                total_steps = int(match.group(2))
                val_loss = float(match.group(3))
                steps.append(step)
                val_losses.append(val_loss)
    
    return steps, val_losses


def plot_loss(steps, val_losses, output_file=None, title=None):
    """Plot validation loss over training steps."""
    if not steps:
        print("No loss data found in log file!")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, val_losses, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title(title or 'Validation Loss Over Training', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot validation loss from training log')
    parser.add_argument('log_file', type=str, help='Path to log file (e.g., logs/run_id.txt)')
    parser.add_argument('--output', '-o', type=str, default=None, 
                       help='Output file path for the plot (e.g., loss_plot.png). If not specified, displays interactively.')
    parser.add_argument('--title', '-t', type=str, default=None,
                       help='Title for the plot')
    
    args = parser.parse_args()
    
    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: Log file not found: {args.log_file}")
        sys.exit(1)
    
    print(f"Parsing log file: {log_path}")
    steps, val_losses = parse_log_file(log_path)
    
    if not steps:
        print("No validation loss data found in the log file.")
        print("Make sure the training has run validation steps.")
        sys.exit(1)
    
    print(f"Found {len(steps)} validation loss measurements")
    print(f"Steps: {min(steps)} to {max(steps)}")
    print(f"Loss range: {min(val_losses):.4f} to {max(val_losses):.4f}")
    
    plot_loss(steps, val_losses, args.output, args.title)


if __name__ == '__main__':
    main()

