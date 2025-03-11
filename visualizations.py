"""
Visualization module for portfolio analysis.

This module contains functions for creating visualizations for portfolio analysis,
including efficient frontier, asset allocation, and performance charts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_efficient_frontier(returns, volatilities, sharpe_ratios=None, 
                           max_sharpe_point=None, min_volatility_point=None,
                           assets_returns=None, assets_volatilities=None, 
                           assets_names=None, monte_carlo_results=None,
                           title="Efficient Frontier", save_path=None):
    """
    Plot the efficient frontier with optional additional elements
    
    Parameters:
    -----------
    returns : array-like
        Returns for each point on the efficient frontier
    volatilities : array-like
        Volatilities for each point on the efficient frontier
    sharpe_ratios : array-like, optional
        Sharpe ratios for each point on the efficient frontier
    max_sharpe_point : tuple, optional
        (volatility, return) for maximum Sharpe ratio portfolio
    min_volatility_point : tuple, optional
        (volatility, return) for minimum volatility portfolio
    assets_returns : array-like, optional
        Returns for individual assets
    assets_volatilities : array-like, optional
        Volatilities for individual assets
    assets_names : array-like, optional
        Names of individual assets
    monte_carlo_results : tuple, optional
        (volatilities, returns) for Monte Carlo simulations
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    # Set up the plot
    fig = plt.figure(figsize=(12, 8))
    
    # Plot the efficient frontier
    if sharpe_ratios is not None:
        plt.scatter(volatilities, returns, c=sharpe_ratios, cmap='viridis', 
                   marker='o', s=10, alpha=0.8)
        cbar = plt.colorbar()
        cbar.set_label('Sharpe Ratio')
    else:
        plt.plot(volatilities, returns, 'b-', linewidth=3)
    
    # Plot the maximum Sharpe ratio portfolio
    if max_sharpe_point is not None:
        plt.scatter(max_sharpe_point[0], max_sharpe_point[1], marker='*', 
                   color='r', s=300, label='Maximum Sharpe Ratio')
    
    # Plot the minimum volatility portfolio
    if min_volatility_point is not None:
        plt.scatter(min_volatility_point[0], min_volatility_point[1], marker='*', 
                   color='g', s=300, label='Minimum Volatility')
    
    # Plot individual assets
    if assets_returns is not None and assets_volatilities is not None:
        if assets_names is not None:
            for i, name in enumerate(assets_names):
                plt.scatter(assets_volatilities[i], assets_returns[i], marker='o', 
                           color='black', s=100, label=name)
        else:
            plt.scatter(assets_volatilities, assets_returns, marker='o',
                       color='black', s=100, label='Assets')
    
    # Plot Monte Carlo simulation results
    if monte_carlo_results is not None:
        mc_vols, mc_returns = monte_carlo_results
        plt.scatter(mc_vols, mc_returns, c='gray', alpha=0.1, s=5)
    
    # Add labels and title
    plt.xlabel('Annualized Volatility')
    plt.ylabel('Annualized Return')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Add legend if we have individual points
    if (max_sharpe_point is not None or min_volatility_point is not None or 
        assets_returns is not None):
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save the figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_allocation(weights, labels, title="Portfolio Allocation", save_path=None):
    """
    Plot portfolio allocation as a pie chart
    
    Parameters:
    -----------
    weights : array-like
        Portfolio weights
    labels : array-like
        Asset names
    title : str, optional
        Chart title
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    # Normalize weights
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    
    # Create a dictionary of weights and labels
    allocation = dict(zip(labels, weights))
    
    # Sort by weight and filter out small weights
    allocation = dict(sorted(allocation.items(), key=lambda x: x[1], reverse=True))
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        allocation.values(),
        labels=allocation.keys(),
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops={'edgecolor': 'w', 'linewidth': 1}
    )
    
    # Style the chart
    ax.set_title(title, fontsize=16)
    plt.setp(autotexts, size=10, weight="bold")
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    
    # Save the figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_performance_over_time(returns, benchmark_returns=None, 
                             title="Portfolio Performance", save_path=None):
    """
    Plot cumulative performance over time
    
    Parameters:
    -----------
    returns : pd.Series
        Portfolio returns time series
    benchmark_returns : pd.Series, optional
        Benchmark returns time series
    title : str, optional
        Chart title
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    # Convert returns to cumulative returns
    cumulative_returns = (1 + returns).cumprod()
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot portfolio performance
    ax.plot(cumulative_returns.index, cumulative_returns, 
           linewidth=2, label='Portfolio')
    
    # Plot benchmark if provided
    if benchmark_returns is not None:
        cumulative_benchmark = (1 + benchmark_returns).cumprod()
        ax.plot(cumulative_benchmark.index, cumulative_benchmark, 
               linewidth=2, linestyle='--', label='Benchmark')
    
    # Add labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend()
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y-1)))
    
    plt.tight_layout()
    
    # Save the figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_drawdowns(returns, top_n=5, title="Portfolio Drawdowns", save_path=None):
    """
    Plot the top N drawdowns
    
    Parameters:
    -----------
    returns : pd.Series
        Portfolio returns time series
    top_n : int, optional
        Number of top drawdowns to plot
    title : str, optional
        Chart title
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    # Calculate cumulative returns
    cumulative_returns = (1 + returns).cumprod()
    
    # Calculate running maximum
    running_max = cumulative_returns.cummax()
    
    # Calculate drawdowns
    drawdown = (cumulative_returns - running_max) / running_max
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot all drawdowns
    ax.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
    
    # Find and highlight top drawdowns
    underwater = drawdown < 0
    underwater_periods = []
    current_period = None
    
    for i, (date, value) in enumerate(drawdown.items()):
        if value < 0 and current_period is None:
            # Start of underwater period
            current_period = {'start': date, 'underwater': []}
        
        if value < 0 and current_period is not None:
            # During underwater period
            current_period['underwater'].append((date, value))
        
        if (value >= 0 or i == len(drawdown) - 1) and current_period is not None:
            # End of underwater period
            current_period['end'] = date
            current_period['max_drawdown'] = min([x[1] for x in current_period['underwater']])
            underwater_periods.append(current_period)
            current_period = None
    
    # Sort underwater periods by maximum drawdown
    underwater_periods.sort(key=lambda x: x['max_drawdown'])
    
    # Plot top N drawdowns
    colors = plt.cm.viridis(np.linspace(0, 0.8, min(top_n, len(underwater_periods))))
    
    for i, period in enumerate(underwater_periods[:top_n]):
        start = period['start']
        end = period['end']
        max_dd = period['max_drawdown']
        
        # Get subset of drawdown series for this period
        period_drawdown = drawdown.loc[start:end]
        
        # Plot this drawdown period
        ax.plot(period_drawdown.index, period_drawdown, 
               linewidth=2, color=colors[i], 
               label=f'Drawdown {i+1}: {max_dd:.1%}')
    
    # Add labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend()
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    plt.tight_layout()
    
    # Save the figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_rolling_metrics(returns, window=252, benchmark_returns=None,
                        metrics=['return', 'volatility', 'sharpe'],
                        title="Rolling Performance Metrics", save_path=None):
    """
    Plot rolling performance metrics
    
    Parameters:
    -----------
    returns : pd.Series
        Portfolio returns time series
    window : int, optional
        Rolling window size
    benchmark_returns : pd.Series, optional
        Benchmark returns time series
    metrics : list, optional
        List of metrics to plot
    title : str, optional
        Chart title
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    # Calculate rolling metrics
    rolling_data = {}
    
    if 'return' in metrics:
        rolling_data['Rolling Return'] = returns.rolling(window=window).mean() * window
        if benchmark_returns is not None:
            rolling_data['Benchmark Return'] = benchmark_returns.rolling(window=window).mean() * window
    
    if 'volatility' in metrics:
        rolling_data['Rolling Volatility'] = returns.rolling(window=window).std() * np.sqrt(window)
        if benchmark_returns is not None:
            rolling_data['Benchmark Volatility'] = benchmark_returns.rolling(window=window).std() * np.sqrt(window)
    
    if 'sharpe' in metrics:
        rolling_return = returns.rolling(window=window).mean() * window
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(window)
        rolling_data['Rolling Sharpe'] = rolling_return / rolling_vol
        if benchmark_returns is not None:
            bench_return = benchmark_returns.rolling(window=window).mean() * window
            bench_vol = benchmark_returns.rolling(window=window).std() * np.sqrt(window)
            rolling_data['Benchmark Sharpe'] = bench_return / bench_vol
    
    # Create a DataFrame from the rolling data
    rolling_df = pd.DataFrame(rolling_data)
    
    # Create the figure with subplots for each metric
    num_metrics = len(metrics)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 4 * num_metrics), sharex=True)
    
    # If there's only one metric, make axes iterable
    if num_metrics == 1:
        axes = [axes]
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        if metric == 'return':
            rolling_df['Rolling Return'].plot(ax=ax, linewidth=2)
            if benchmark_returns is not None:
                rolling_df['Benchmark Return'].plot(ax=ax, linewidth=2, linestyle='--')
            ax.set_title(f'Rolling {window}-day Annualized Return')
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        elif metric == 'volatility':
            rolling_df['Rolling Volatility'].plot(ax=ax, linewidth=2)
            if benchmark_returns is not None:
                rolling_df['Benchmark Volatility'].plot(ax=ax, linewidth=2, linestyle='--')
            ax.set_title(f'Rolling {window}-day Annualized Volatility')
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        elif metric == 'sharpe':
            rolling_df['Rolling Sharpe'].plot(ax=ax, linewidth=2)
            if benchmark_returns is not None:
                rolling_df['Benchmark Sharpe'].plot(ax=ax, linewidth=2, linestyle='--')
            ax.set_title(f'Rolling {window}-day Sharpe Ratio')
            
            # Add a horizontal line at 0
            ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        ax.grid(True, alpha=0.3)
        
        if benchmark_returns is not None:
            ax.legend(['Portfolio', 'Benchmark'])
        else:
            ax.legend(['Portfolio'])
    
    # Add overall title
    fig.suptitle(title, fontsize=16, y=1.02)
    
    plt.tight_layout()
    
    # Save the figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig