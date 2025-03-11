"""
Market scenario simulation module for portfolio analysis.

This module contains functions for simulating portfolio performance under
different market scenarios using Monte Carlo methods.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def simulate_portfolio_performance(weights, mean_returns, cov_matrix, 
                                 initial_investment=10000, periods=252, simulations=1000,
                                 return_scaling=1.0, vol_scaling=1.0):
    """
    Simulate portfolio performance using Monte Carlo simulation
    
    Parameters:
    -----------
    weights : array-like
        Portfolio weights
    mean_returns : array-like
        Expected returns for each asset
    cov_matrix : array-like
        Covariance matrix of returns
    initial_investment : float, optional
        Initial investment amount
    periods : int, optional
        Number of periods to simulate
    simulations : int, optional
        Number of simulations to run
    return_scaling : float, optional
        Scaling factor for returns (for scenario testing)
    vol_scaling : float, optional
        Scaling factor for volatility (for scenario testing)
        
    Returns:
    --------
    tuple : (cumulative_returns, simulation_results)
    """
    # Ensure weights are normalized
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    
    # Adjust returns and volatility for scenario
    adjusted_returns = mean_returns * return_scaling / 252  # Daily returns
    adjusted_cov = cov_matrix * vol_scaling / 252  # Daily covariance
    
    # Generate random returns
    random_returns = np.random.multivariate_normal(
        adjusted_returns,
        adjusted_cov,
        (simulations, periods)
    )
    
    # Calculate portfolio returns for each simulation
    portfolio_returns = np.dot(random_returns, weights)
    
    # Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + portfolio_returns, axis=1) * initial_investment
    
    # Calculate simulation results
    simulation_results = {
        'median': np.median(cumulative_returns[:, -1]),
        'mean': np.mean(cumulative_returns[:, -1]),
        'std': np.std(cumulative_returns[:, -1]),
        'min': np.min(cumulative_returns[:, -1]),
        'max': np.max(cumulative_returns[:, -1]),
        'percentile_5': np.percentile(cumulative_returns[:, -1], 5),
        'percentile_95': np.percentile(cumulative_returns[:, -1], 95),
        'max_drawdown': calculate_max_drawdown(cumulative_returns),
        'positive_return_prob': np.mean(cumulative_returns[:, -1] > initial_investment)
    }
    
    return cumulative_returns, simulation_results


def calculate_max_drawdown(cumulative_returns):
    """
    Calculate the average maximum drawdown across all simulations
    
    Parameters:
    -----------
    cumulative_returns : array-like
        Cumulative returns for each simulation
        
    Returns:
    --------
    float : Average maximum drawdown
    """
    max_drawdowns = []
    
    for path in cumulative_returns:
        # Calculate the running maximum
        running_max = np.maximum.accumulate(path)
        
        # Calculate the drawdown
        drawdown = (path - running_max) / running_max
        
        # Store the maximum drawdown
        max_drawdowns.append(np.min(drawdown))
    
    return np.mean(max_drawdowns)


def simulate_market_scenarios(weights, mean_returns, cov_matrix, scenarios=None,
                            initial_investment=10000, years=5, simulations=1000):
    """
    Simulate portfolio performance under different market scenarios
    
    Parameters:
    -----------
    weights : array-like
        Portfolio weights
    mean_returns : array-like
        Expected returns for each asset
    cov_matrix : array-like
        Covariance matrix of returns
    scenarios : dict, optional
        Dictionary of scenario names and adjustments
    initial_investment : float, optional
        Initial investment amount
    years : int, optional
        Number of years to simulate
    simulations : int, optional
        Number of simulations per scenario
        
    Returns:
    --------
    tuple : (scenario_results, fig)
    """
    # Define default scenarios if none provided
    if scenarios is None:
        scenarios = {
            'Base Case': {'return_mult': 1.0, 'vol_mult': 1.0},
            'Bull Market': {'return_mult': 1.5, 'vol_mult': 1.1},
            'Bear Market': {'return_mult': 0.7, 'vol_mult': 1.3},
            'High Volatility': {'return_mult': 1.1, 'vol_mult': 1.8},
            'Low Volatility': {'return_mult': 0.9, 'vol_mult': 0.7},
            'Stagflation': {'return_mult': 0.8, 'vol_mult': 1.5},
            'Recovery': {'return_mult': 1.3, 'vol_mult': 1.2},
            'Financial Crisis': {'return_mult': 0.5, 'vol_mult': 2.0}
        }
    
    # Calculate number of periods
    periods = years * 252  # Daily steps
    
    # Container for results
    scenario_results = {}
    
    # Create figure for plots
    fig, axs = plt.subplots(2, (len(scenarios) + 1) // 2, 
                           figsize=(15, 10), squeeze=False)
    axs = axs.flatten()
    
    # Simulate each scenario
    for i, (name, adjustment) in enumerate(scenarios.items()):
        # Run simulation
        cumulative_returns, results = simulate_portfolio_performance(
            weights, mean_returns, cov_matrix,
            initial_investment=initial_investment,
            periods=periods,
            simulations=simulations,
            return_scaling=adjustment['return_mult'],
            vol_scaling=adjustment['vol_mult']
        )
        
        # Store results
        scenario_results[name] = results
        
        # Plot this scenario
        time = np.linspace(0, years, periods)
        median_line = np.median(cumulative_returns, axis=0)
        percentile_5 = np.percentile(cumulative_returns, 5, axis=0)
        percentile_95 = np.percentile(cumulative_returns, 95, axis=0)
        
        axs[i].plot(time, median_line, linewidth=2, color='blue')
        axs[i].fill_between(time, percentile_5, percentile_95, color='blue', alpha=0.2)
        axs[i].set_title(name)
        axs[i].set_xlabel('Years')
        axs[i].set_ylabel('Portfolio Value')
        axs[i].grid(True, alpha=0.3)
        
        # Add final portfolio value and probability
        axs[i].text(
            0.05, 0.95,
            f"Final Value (median): ${scenario_results[name]['median']:,.0f}\n"
            f"Probability > Initial: {scenario_results[name]['positive_return_prob']:.1%}",
            transform=axs[i].transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    
    # Hide any unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')
    
    plt.tight_layout()
    
    return scenario_results, fig


def generate_stress_test_report(weights, mean_returns, cov_matrix, stress_scenarios=None,
                              initial_investment=10000, periods=252, simulations=1000):
    """
    Generate a stress test report for a portfolio
    
    Parameters:
    -----------
    weights : array-like
        Portfolio weights
    mean_returns : array-like
        Expected returns for each asset
    cov_matrix : array-like
        Covariance matrix of returns
    stress_scenarios : dict, optional
        Dictionary of stress scenario names and adjustments
    initial_investment : float, optional
        Initial investment amount
    periods : int, optional
        Number of periods to simulate
    simulations : int, optional
        Number of simulations per scenario
        
    Returns:
    --------
    tuple : (stress_results, fig)
    """
    # Define default stress scenarios if none provided
    if stress_scenarios is None:
        stress_scenarios = {
            'Market Crash': {'return_mult': 0.3, 'vol_mult': 2.5},
            'Severe Recession': {'return_mult': 0.5, 'vol_mult': 2.0},
            'Stagflation Crisis': {'return_mult': 0.6, 'vol_mult': 1.8},
            'Liquidity Crisis': {'return_mult': 0.7, 'vol_mult': 2.2},
            'Interest Rate Shock': {'return_mult': 0.8, 'vol_mult': 1.6}
        }
    
    # Container for results
    stress_results = {}
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Simulate each stress scenario
    for name, adjustment in stress_scenarios.items():
        # Run simulation
        _, results = simulate_portfolio_performance(
            weights, mean_returns, cov_matrix,
            initial_investment=initial_investment,
            periods=periods,
            simulations=simulations,
            return_scaling=adjustment['return_mult'],
            vol_scaling=adjustment['vol_mult']
        )
        
        # Store results
        stress_results[name] = results
    
    # Create a DataFrame for display
    results_df = pd.DataFrame({
        scenario: {
            'Expected Loss (%)': (results['median'] - initial_investment) / initial_investment * 100,
            'Worst-Case Loss (%)': (results['percentile_5'] - initial_investment) / initial_investment * 100,
            'Maximum Drawdown (%)': results['max_drawdown'] * 100,
            'Probability of Loss (%)': (1 - results['positive_return_prob']) * 100
        }
        for scenario, results in stress_results.items()
    }).T
    
    # Sort by expected loss
    results_df = results_df.sort_values('Expected Loss (%)')
    
    # Plot the results
    colors = plt.cm.RdYlGn(np.linspace(0, 0.7, len(results_df)))
    results_df[['Expected Loss (%)', 'Worst-Case Loss (%)']].plot(
        kind='barh', ax=ax, color=colors)
    
    # Style the plot
    ax.set_title('Portfolio Stress Test Results', fontsize=14)
    ax.set_xlabel('Loss (%)')
    ax.grid(True, axis='x', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.2)
    
    # Add annotations
    for i, v in enumerate(results_df['Expected Loss (%)']):
        ax.text(v - 1, i, f'{v:.1f}%', va='center')
    
    for i, v in enumerate(results_df['Worst-Case Loss (%)']):
        ax.text(v - 1, i, f'{v:.1f}%', va='center')
    
    plt.tight_layout()
    
    return stress_results, fig, results_df
