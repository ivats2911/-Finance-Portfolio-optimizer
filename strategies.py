"""
Portfolio optimization strategies module.

This module contains functions for implementing various portfolio optimization
strategies, including maximum Sharpe ratio, minimum volatility, and efficient frontier.
"""

import numpy as np
from scipy.optimize import minimize
from risk_metrics import portfolio_performance


def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate=0.01):
    """
    Calculate the weights for a portfolio with maximum Sharpe ratio
    
    Parameters:
    -----------
    mean_returns : array-like
        Expected returns for each asset
    cov_matrix : array-like
        Covariance matrix of returns
    risk_free_rate : float, optional
        Risk-free rate
        
    Returns:
    --------
    dict : Optimization results
    """
    num_assets = len(mean_returns)
    
    # Function to be minimized (negative of Sharpe ratio)
    def neg_sharpe_ratio(weights):
        return -portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[2]
    
    # Constraints and bounds
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    initial_guess = num_assets * [1. / num_assets]
    
    # Optimize
    result = minimize(neg_sharpe_ratio, initial_guess, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    # Get results
    optimal_weights = result['x']
    performance = portfolio_performance(optimal_weights, mean_returns, cov_matrix, risk_free_rate)
    
    return {
        'weights': optimal_weights,
        'performance': performance,
        'success': result['success']
    }


def min_volatility(mean_returns, cov_matrix):
    """
    Calculate the weights for a minimum volatility portfolio
    
    Parameters:
    -----------
    mean_returns : array-like
        Expected returns for each asset
    cov_matrix : array-like
        Covariance matrix of returns
        
    Returns:
    --------
    dict : Optimization results
    """
    num_assets = len(mean_returns)
    
    # Function to be minimized (portfolio volatility)
    def portfolio_volatility(weights):
        return portfolio_performance(weights, mean_returns, cov_matrix)[1]
    
    # Constraints and bounds
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    initial_guess = num_assets * [1. / num_assets]
    
    # Optimize
    result = minimize(portfolio_volatility, initial_guess, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    # Get results
    optimal_weights = result['x']
    performance = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
    
    return {
        'weights': optimal_weights,
        'performance': performance,
        'success': result['success']
    }


def max_return_given_risk(mean_returns, cov_matrix, target_volatility):
    """
    Calculate the weights for a portfolio with maximum return given a target volatility
    
    Parameters:
    -----------
    mean_returns : array-like
        Expected returns for each asset
    cov_matrix : array-like
        Covariance matrix of returns
    target_volatility : float
        Target portfolio volatility
        
    Returns:
    --------
    dict : Optimization results
    """
    num_assets = len(mean_returns)
    
    # Function to be maximized (portfolio return)
    def neg_portfolio_return(weights):
        return -portfolio_performance(weights, mean_returns, cov_matrix)[0]
    
    # Constraints and bounds
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'eq', 'fun': lambda x: portfolio_performance(weights, mean_returns, cov_matrix)[1] - target_volatility}
    )
    bounds = tuple((0, 1) for asset in range(num_assets))
    initial_guess = num_assets * [1. / num_assets]
    
    # Optimize
    result = minimize(neg_portfolio_return, initial_guess, method='SLSQP',
                     bounds=bounds, constraints=constraints)
    
    # Get results
    optimal_weights = result['x']
    performance = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
    
    return {
        'weights': optimal_weights,
        'performance': performance,
        'success': result['success']
    }


def min_risk_given_return(mean_returns, cov_matrix, target_return):
    """
    Calculate the weights for a portfolio with minimum risk given a target return
    
    Parameters:
    -----------
    mean_returns : array-like
        Expected returns for each asset
    cov_matrix : array-like
        Covariance matrix of returns
    target_return : float
        Target portfolio return
        
    Returns:
    --------
    dict : Optimization results
    """
    num_assets = len(mean_returns)
    
    # Function to be minimized (portfolio volatility)
    def portfolio_volatility(weights):
        return portfolio_performance(weights, mean_returns, cov_matrix)[1]
    
    # Constraints and bounds
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'eq', 'fun': lambda x: portfolio_performance(x, mean_returns, cov_matrix)[0] - target_return}
    )
    bounds = tuple((0, 1) for asset in range(num_assets))
    initial_guess = num_assets * [1. / num_assets]
    
    # Optimize
    result = minimize(portfolio_volatility, initial_guess, method='SLSQP',
                     bounds=bounds, constraints=constraints)
    
    # Get results
    optimal_weights = result['x']
    performance = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
    
    return {
        'weights': optimal_weights,
        'performance': performance,
        'success': result['success']
    }


def efficient_frontier(mean_returns, cov_matrix, points=50):
    """
    Calculate the efficient frontier
    
    Parameters:
    -----------
    mean_returns : array-like
        Expected returns for each asset
    cov_matrix : array-like
        Covariance matrix of returns
    points : int, optional
        Number of points on the efficient frontier
        
    Returns:
    --------
    tuple : (returns, volatilities, sharpe_ratios, weights_array)
    """
    # Get minimum volatility portfolio
    min_vol_results = min_volatility(mean_returns, cov_matrix)
    min_vol_return = min_vol_results['performance'][0]
    min_vol_weights = min_vol_results['weights']
    
    # Get maximum return (single asset with highest return)
    max_return_idx = np.argmax(mean_returns)
    max_return = mean_returns[max_return_idx]
    
    # Create a range of target returns
    target_returns = np.linspace(min_vol_return, max_return, points)
    efficient_weights = []
    efficient_volatilities = []
    
    # For each target return, find the minimum volatility portfolio
    for target in target_returns:
        result = min_risk_given_return(mean_returns, cov_matrix, target)
        
        if result['success']:
            efficient_weights.append(result['weights'])
            efficient_volatilities.append(result['performance'][1])
        else:
            # If optimization fails, use the last successful result
            efficient_weights.append(efficient_weights[-1] if efficient_weights else min_vol_weights)
            efficient_volatilities.append(efficient_volatilities[-1] if efficient_volatilities else min_vol_results['performance'][1])
    
    # Calculate Sharpe ratios (assumed risk-free rate of 0.01)
    risk_free_rate = 0.01
    efficient_sharpe = [(target - risk_free_rate) / vol for target, vol in 
                       zip(target_returns, efficient_volatilities)]
    
    return target_returns, efficient_volatilities, efficient_sharpe, efficient_weights


def max_diversification(mean_returns, cov_matrix):
    """
    Calculate weights for a maximum diversification portfolio
    
    Parameters:
    -----------
    mean_returns : array-like
        Expected returns for each asset
    cov_matrix : array-like
        Covariance matrix of returns
        
    Returns:
    --------
    dict : Optimization results
    """
    num_assets = len(mean_returns)
    asset_vols = np.sqrt(np.diag(cov_matrix))
    
    # Function to be minimized (negative of diversification ratio)
    def neg_diversification_ratio(weights):
        weights = np.array(weights)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        weighted_vol_sum = np.sum(weights * asset_vols)
        return -weighted_vol_sum / port_vol
    
    # Constraints and bounds
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    initial_guess = num_assets * [1. / num_assets]
    
    # Optimize
    result = minimize(neg_diversification_ratio, initial_guess, method='SLSQP',
                     bounds=bounds, constraints=constraints)
    
    # Get results
    optimal_weights = result['x']
    performance = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
    
    return {
        'weights': optimal_weights,
        'performance': performance,
        'success': result['success']
    }


def risk_parity(cov_matrix):
    """
    Calculate weights for a risk parity portfolio
    
    Parameters:
    -----------
    cov_matrix : array-like
        Covariance matrix of returns
        
    Returns:
    --------
    dict : Optimization results
    """
    num_assets = cov_matrix.shape[0]
    
    # Function to be minimized (variance of risk contributions)
    def risk_parity_objective(weights):
        weights = np.array(weights)
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Portfolio risk
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Risk contribution of each asset
        marginal_risk = np.dot(cov_matrix, weights)
        risk_contribution = weights * marginal_risk / port_vol
        
        # Target risk contribution (equal for all assets)
        target_risk = port_vol / num_assets
        
        # Sum of squared differences between actual and target risk contributions
        return np.sum((risk_contribution - target_risk) ** 2)
    
    # Constraints and bounds
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.01, 1) for asset in range(num_assets))  # Lower bound > 0 to avoid division by zero
    initial_guess = num_assets * [1. / num_assets]
    
    # Optimize
    result = minimize(risk_parity_objective, initial_guess, method='SLSQP',
                     bounds=bounds, constraints=constraints)
    
    # Get results
    optimal_weights = result['x']
    optimal_weights = optimal_weights / np.sum(optimal_weights)  # Normalize
    
    return {
        'weights': optimal_weights,
        'success': result['success']
    }
