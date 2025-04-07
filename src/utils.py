"""Utility functions. Honestly not really utilized..."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Data saved to {filename}")

def load_json(filename):
    """Load data from a JSON file"""
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return None
        
    with open(filename, 'r') as f:
        return json.load(f)

def plot_elbow_curve(k_range, wcss, optimal_k=None):
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, wcss, 'o-', markersize=8, markerfacecolor='blue', markeredgecolor='black')
    
    if optimal_k is not None:
        plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
    
    plt.title('Finding the optimal number of clusters using the Elbow Method', fontsize=16)
    plt.xlabel('Number of clusters (k)', fontsize=14)
    plt.ylabel('WCSS', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # Save to file
    plt.savefig('elbow_curve.png', dpi=300, bbox_inches='tight')
    
    # Display
    plt.show()

def plot_score_distribution(filtered_songs, score_column, bins=10):
    plt.figure(figsize=(10, 6))
    
    plt.hist(filtered_songs[score_column], bins=bins, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {score_column}', fontsize=16)
    plt.xlabel('Score', fontsize=14)
    plt.ylabel('Number of Songs', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # mean line
    mean_score = filtered_songs[score_column].mean()
    plt.axvline(x=mean_score, color='r', linestyle='--', 
                label=f'Mean: {mean_score:.2f}')
    
    plt.legend()
    plt.tight_layout()
    
    # Save to file
    plt.savefig(f'{score_column}_distribution.png', dpi=300, bbox_inches='tight')
    
    # Display
    plt.show()