"""
Risk metrics module for portfolio analysis.

This module contains functions for calculating various risk and performance metrics
for portfolios, including Sharpe ratio, maximum drawdown, Value at Risk, etc.
"""

import numpy as np
import pandas as pd


def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    """
    Calculate portfolio performance metrics
    
    Parameters:
    -----------
    weights : array-like
        Asset allocation weights
    mean_returns : array-like
        Expected returns for each asset
    cov_matrix : array-like
        Covariance matrix of returns
    risk_free_rate : float, optional
        Risk-free rate
        
    Returns:
    --------
    tuple : (return, volatility, sharpe_ratio)
    """
    # Ensure weights are normalized
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    
    # Expected portfolio return
    portfolio_return = np.sum(mean_returns * weights)
    
    # Expected portfolio volatility
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Sharpe ratio
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    
    return portfolio_return, portfolio_volatility, sharpe_ratio


def calculate_max_drawdown(returns):
    """
    Calculate the maximum drawdown from a series of returns
    
    Parameters:
    -----------
    returns : array-like
        Series of returns
        
    Returns:
    --------
    float : Maximum drawdown
    """
    # Convert returns to cumulative returns
    cum_returns = (1 + returns).cumprod()
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(cum_returns)
    
    # Calculate drawdown
    drawdown = (cum_returns - running_max) / running_max
    
    # Get maximum drawdown
    max_drawdown = drawdown.min()
    
    return max_drawdown


def calculate_var(returns, confidence_level=0.05):
    """
    Calculate Value at Risk (VaR)
    
    Parameters:
    -----------
    returns : array-like
        Series of returns
    confidence_level : float, optional
        Confidence level for VaR (default: 0.05 for 95% VaR)
        
    Returns:
    --------
    float : Value at Risk
    """
    # Calculate VaR as the negative of the return at the specified percentile
    var = -np.percentile(returns, confidence_level * 100)
    
    return var


def calculate_cvar(returns, confidence_level=0.05):
    """
    Calculate Conditional Value at Risk (CVaR) / Expected Shortfall
    
    Parameters:
    -----------
    returns : array-like
        Series of returns
    confidence_level : float, optional
        Confidence level for CVaR (default: 0.05 for 95% CVaR)
        
    Returns:
    --------
    float : Conditional Value at Risk
    """
    # Calculate VaR
    var = calculate_var(returns, confidence_level)
    
    # Filter returns beyond VaR
    beyond_var = returns[returns <= -var]
    
    # Calculate CVaR as the mean of returns beyond VaR
    cvar = -beyond_var.mean()
    
    return cvar


def calculate_sortino_ratio(returns, risk_free_rate=0.01, periods_per_year=252):
    """
    Calculate Sortino ratio (using downside deviation)
    
    Parameters:
    -----------
    returns : array-like
        Series of returns
    risk_free_rate : float, optional
        Risk-free rate
    periods_per_year : int, optional
        Number of periods per year for annualization
        
    Returns:
    --------
    float : Sortino ratio
    """
    # Calculate excess returns
    excess_returns = np.mean(returns) * periods_per_year - risk_free_rate
    
    # Calculate downside deviation (standard deviation of negative returns)
    downside_returns = returns[returns < 0]
    downside_deviation = np.std(downside_returns) * np.sqrt(periods_per_year)
    
    # Calculate Sortino ratio
    sortino_ratio = excess_returns / downside_deviation
    
    return sortino_ratio


def calculate_calmar_ratio(returns, periods_per_year=252):
    """
    Calculate Calmar ratio (return / maximum drawdown)
    
    Parameters:
    -----------
    returns : array-like
        Series of returns
    periods_per_year : int, optional
        Number of periods per year for annualization
        
    Returns:
    --------
    float : Calmar ratio
    """
    # Calculate annualized return
    annualized_return = np.mean(returns) * periods_per_year
    
    # Calculate maximum drawdown
    max_drawdown = calculate_max_drawdown(returns)
    
    # Calculate Calmar ratio
    calmar_ratio = annualized_return / abs(max_drawdown)
    
    return calmar_ratio


def generate_risk_report(weights, returns, mean_returns, cov_matrix, 
                        risk_free_rate=0.01, periods_per_year=252):
    """
    Generate a comprehensive risk report for a portfolio
    
    Parameters:
    -----------
    weights : array-like
        Asset allocation weights
    returns : pd.DataFrame
        Historical returns
    mean_returns : array-like
        Expected returns for each asset
    cov_matrix : array-like
        Covariance matrix of returns
    risk_free_rate : float, optional
        Risk-free rate
    periods_per_year : int, optional
        Number of periods per year for annualization
        
    Returns:
    --------
    dict : Risk metrics
    """
    # Calculate portfolio returns
    portfolio_returns = np.dot(returns, weights)
    
    # Calculate basic performance metrics
    return_annual, volatility_annual, sharpe = portfolio_performance(
        weights, mean_returns, cov_matrix, risk_free_rate)
    
    # Calculate other risk metrics
    max_drawdown = calculate_max_drawdown(portfolio_returns)
    var_95 = calculate_var(portfolio_returns, 0.05)
    var_99 = calculate_var(portfolio_returns, 0.01)
    cvar_95 = calculate_cvar(portfolio_returns, 0.05)
    sortino = calculate_sortino_ratio(portfolio_returns, risk_free_rate, periods_per_year)
    calmar = calculate_calmar_ratio(portfolio_returns, periods_per_year)
    
    # Calculate return statistics
    best_return = portfolio_returns.max()
    worst_return = portfolio_returns.min()
    positive_days = np.mean(portfolio_returns > 0)
    
    # Create report
    report = {
        'Annual Return': return_annual,
        'Annual Volatility': volatility_annual,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Calmar Ratio': calmar,
        'Maximum Drawdown': max_drawdown,
        'Value at Risk (95%)': var_95,
        'Value at Risk (99%)': var_99,
        'Conditional VaR (95%)': cvar_95,
        'Best Return': best_return,
        'Worst Return': worst_return,
        'Positive Returns Ratio': positive_days
    }
    
    return report
