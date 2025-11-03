"""
Sensitivity analysis and explainability for quantum models.

This module provides tools for analyzing parameter sensitivity and
understanding quantum model behavior.
"""

import numpy as np
import matplotlib.pyplot as plt


def sensitivity_analyzer(params, noise_impact=False):
    """
    Analyze and visualize parameter sensitivity.
    
    Creates a plot showing the sensitivity of different parameters,
    which helps understand model behavior and identify critical parameters.
    
    Args:
        params: Model parameters to analyze (numpy array)
        noise_impact: If True, annotate regions with high noise sensitivity
    
    Returns:
        Confirmation message with plot filename
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot flattened parameters
    params_flat = params.flatten()
    ax.plot(params_flat, 'b-', linewidth=2, label='Parameter Values')
    
    # Style the plot
    ax.set_xlabel('Parameter Index', fontsize=12)
    ax.set_ylabel('Parameter Value', fontsize=12)
    ax.set_title('Parameter Sensitivity Analysis', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add noise impact annotation if requested
    if noise_impact:
        # Find regions with high variance (more sensitive)
        threshold = params_flat.std() * 1.5
        high_sensitivity = np.abs(params_flat) > threshold
        if np.any(high_sensitivity):
            ax.scatter(np.where(high_sensitivity)[0], params_flat[high_sensitivity], 
                      color='red', s=50, alpha=0.7, label='High Noise Impact')
            ax.annotate('High Noise Impact Regions', 
                       xy=(np.argmax(np.abs(params_flat)), params_flat[np.argmax(np.abs(params_flat))]),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.savefig("sensitivity.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return "Sensitivity analysis plot saved as sensitivity.png"
