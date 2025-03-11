"""
Main portfolio optimizer class.

This module contains the main PortfolioOptimizer class that combines data management,
optimization strategies, risk metrics, and visualization functions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from data_manager import DataManager
from strategies import (max_sharpe_ratio, min_volatility, efficient_frontier,
                       max_diversification, risk_parity)
from risk_metrics import (portfolio_performance, generate_risk_report)
from visualizations import (plot_efficient_frontier, plot_allocation,
                          plot_performance_over_time, plot_drawdowns,
                          plot_rolling_metrics)
from market_simulator import (simulate_market_scenarios, generate_stress_test_report)


class PortfolioOptimizer:
    """
    A comprehensive portfolio optimization framework that implements various strategies
    to maximize returns while managing risk based on historical data.
    """
    
    def __init__(self, tickers, start_date=None, end_date=None):
        """
        Initialize the portfolio optimizer with stock tickers and date range.
        
        Parameters:
        -----------
        tickers : list
            List of stock ticker symbols
        start_date : str, optional
            Start date for historical data in 'YYYY-MM-DD' format
        end_date : str, optional
            End date for historical data in 'YYYY-MM-DD' format
        """
        # Initialize data manager
        self.data_manager = DataManager(tickers, start_date, end_date)
        self.tickers = tickers
        self.risk_free_rate = 0.01  # Default risk-free rate
        
        # Initialize data containers
        self.data = None
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None
        
    def fetch_data(self):
        """
        Fetch historical price data for the specified tickers
        
        Returns:
        --------
        pd.DataFrame : Historical price data
        """
        self.data = self.data_manager.fetch_data()
        return self.data
    
    def calculate_returns(self, frequency='daily'):
        """
        Calculate returns based on specified frequency
        
        Parameters:
        -----------
        frequency : str
            'daily', 'monthly', or 'annual'
            
        Returns:
        --------
        pd.DataFrame : Returns data
        """
        self.returns, self.mean_returns, self.cov_matrix = self.data_manager.calculate_returns(frequency)
        return self.returns
    
    def set_risk_free_rate(self, rate):
        """
        Set the risk-free rate
        
        Parameters:
        -----------
        rate : float
            Risk-free rate as decimal
        """
        self.risk_free_rate = rate
    
    def portfolio_performance(self, weights):
        """
        Calculate portfolio performance metrics
        
        Parameters:
        -----------
        weights : array-like
            Asset allocation weights
            
        Returns:
        --------
        tuple : (return, volatility, sharpe_ratio)
        """
        return portfolio_performance(weights, self.mean_returns, self.cov_matrix, self.risk_free_rate)
    
    def optimal_portfolio(self, strategy='max_sharpe'):
        """
        Calculate optimal portfolio weights based on different strategies
        
        Parameters:
        -----------
        strategy : str
            'max_sharpe' : Maximize Sharpe ratio
            'min_volatility' : Minimize volatility
            'max_diversification' : Maximize diversification ratio
            'risk_parity' : Equal risk contribution
            
        Returns:
        --------
        dict : Optimization results
        """
        if self.mean_returns is None or self.cov_matrix is None:
            self.calculate_returns()
        
        if strategy == 'max_sharpe':
            result = max_sharpe_ratio(self.mean_returns, self.cov_matrix, self.risk_free_rate)
        elif strategy == 'min_volatility':
            result = min_volatility(self.mean_returns, self.cov_matrix)
        elif strategy == 'max_diversification':
            result = max_diversification(self.mean_returns, self.cov_matrix)
        elif strategy == 'risk_parity':
            result = risk_parity(self.cov_matrix)
            # Add performance metrics for risk parity
            result['performance'] = portfolio_performance(
                result['weights'], self.mean_returns, self.cov_matrix, self.risk_free_rate)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Add allocation dictionary for convenient access
        result['allocation'] = dict(zip(self.tickers, [round(w * 100, 2) for w in result['weights']]))
        
        return result
    
    def calculate_efficient_frontier(self, points=50):
        """
        Calculate the efficient frontier
        
        Parameters:
        -----------
        points : int
            Number of points on the efficient frontier
            
        Returns:
        --------
        tuple : (returns, volatilities, sharpe_ratios, weights_array)
        """
        if self.mean_returns is None or self.cov_matrix is None:
            self.calculate_returns()
        
        return efficient_frontier(self.mean_returns, self.cov_matrix, points)
    
    def visualize_efficient_frontier(self, points=50, show_assets=True, show_monte_carlo=True, 
                                   monte_carlo_simulations=5000, save_path=None):
        """
        Visualize the efficient frontier with optional Monte Carlo simulations
        
        Parameters:
        -----------
        points : int
            Number of points on the efficient frontier
        show_assets : bool
            Whether to show individual assets
        show_monte_carlo : bool
            Whether to show Monte Carlo simulations
        monte_carlo_simulations : int
            Number of Monte Carlo simulations to run
        save_path : str, optional
            Path to save the visualization
            
        Returns:
        --------
        matplotlib.figure.Figure
        """
        # Get efficient frontier
        returns, volatilities, sharpe, weights = self.calculate_efficient_frontier(points)
        
        # Get optimal portfolios
        max_sharpe_portfolio = self.optimal_portfolio('max_sharpe')
        min_vol_portfolio = self.optimal_portfolio('min_volatility')
        
        # Prepare points for optimal portfolios
        max_sharpe_point = (
            max_sharpe_portfolio['performance'][1],  # volatility
            max_sharpe_portfolio['performance'][0]   # return
        )
        
        min_vol_point = (
            min_vol_portfolio['performance'][1],  # volatility
            min_vol_portfolio['performance'][0]   # return
        )
        
        # Prepare individual assets data if requested
        assets_returns = None
        assets_volatilities = None
        assets_names = None
        
        if show_assets:
            assets_returns = self.mean_returns
            assets_volatilities = np.sqrt(np.diag(self.cov_matrix))
            assets_names = self.tickers
        
        # Generate random portfolios for Monte Carlo if requested
        monte_carlo_results = None
        
        if show_monte_carlo:
            mc_vols = []
            mc_returns = []
            
            for _ in range(monte_carlo_simulations):
                # Generate random weights
                weights = np.random.random(len(self.tickers))
                weights = weights / np.sum(weights)
                
                # Calculate performance
                ret, vol, _ = self.portfolio_performance(weights)
                mc_returns.append(ret)
                mc_vols.append(vol)
                
            monte_carlo_results = (mc_vols, mc_returns)
        
        # Create the plot
        fig = plot_efficient_frontier(
            returns, volatilities, sharpe,
            max_sharpe_point=max_sharpe_point,
            min_volatility_point=min_vol_point,
            assets_returns=assets_returns,
            assets_volatilities=assets_volatilities,
            assets_names=assets_names,
            monte_carlo_results=monte_carlo_results,
            title="Portfolio Optimization: Efficient Frontier",
            save_path=save_path
        )
        
        return fig
    
    def visualize_allocation(self, weights, title="Portfolio Allocation", save_path=None):
        """
        Visualize portfolio allocation as a pie chart
        
        Parameters:
        -----------
        weights : array-like
            Portfolio weights
        title : str, optional
            Chart title
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
        """
        return plot_allocation(weights, self.tickers, title, save_path)
    
    def simulate_scenarios(self, weights, scenarios=None, initial_investment=100000, 
                         years=5, simulations=1000, save_path=None):
        """
        Simulate portfolio performance under different market scenarios
        
        Parameters:
        -----------
        weights : array-like
            Portfolio weights
        scenarios : dict, optional
            Dictionary of scenario names and adjustments
        initial_investment : float, optional
            Initial investment amount
        years : int, optional
            Number of years to simulate
        simulations : int, optional
            Number of simulations per scenario
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        tuple : (scenario_results, fig)
        """
        if self.mean_returns is None or self.cov_matrix is None:
            self.calculate_returns()
        
        scenario_results, fig = simulate_market_scenarios(
            weights, self.mean_returns, self.cov_matrix,
            scenarios=scenarios,
            initial_investment=initial_investment,
            years=years,
            simulations=simulations
        )
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return scenario_results, fig
    
    def generate_stress_test(self, weights, stress_scenarios=None, initial_investment=100000,
                          periods=252, simulations=1000, save_path=None):
        """
        Generate a stress test report for a portfolio
        
        Parameters:
        -----------
        weights : array-like
            Portfolio weights
        stress_scenarios : dict, optional
            Dictionary of stress scenario names and adjustments
        initial_investment : float, optional
            Initial investment amount
        periods : int, optional
            Number of periods to simulate
        simulations : int, optional
            Number of simulations per scenario
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        tuple : (stress_results, fig, df)
        """
        if self.mean_returns is None or self.cov_matrix is None:
            self.calculate_returns()
        
        stress_results, fig, df = generate_stress_test_report(
            weights, self.mean_returns, self.cov_matrix,
            stress_scenarios=stress_scenarios,
            initial_investment=initial_investment,
            periods=periods,
            simulations=simulations
        )
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return stress_results, fig, df
    
    def generate_performance_report(self, weights, file_path=None):
        """
        Generate a comprehensive performance report for an optimized portfolio
        
        Parameters:
        -----------
        weights : array-like
            Optimized portfolio weights
        file_path : str, optional
            Path to save the report
            
        Returns:
        --------
        dict : Performance metrics
        """
        if self.returns is None:
            self.calculate_returns()
        
        # Calculate portfolio returns
        portfolio_returns = np.dot(self.returns.values, weights)
        
        # Generate risk report
        report = generate_risk_report(
            weights, self.returns.values, self.mean_returns, 
            self.cov_matrix, self.risk_free_rate
        )
        
        # Add allocation to report
        report['Allocation'] = dict(zip(self.tickers, weights))
        
        if file_path:
            with open(file_path, 'w') as f:
                f.write("PORTFOLIO PERFORMANCE REPORT\n")
                f.write("==========================\n\n")
                
                f.write("PERFORMANCE METRICS:\n")
                f.write(f"Annual Return: {report['Annual Return']:.2%}\n")
                f.write(f"Annual Volatility: {report['Annual Volatility']:.2%}\n")
                f.write(f"Sharpe Ratio: {report['Sharpe Ratio']:.2f}\n")
                f.write(f"Sortino Ratio: {report['Sortino Ratio']:.2f}\n")
                f.write(f"Calmar Ratio: {report['Calmar Ratio']:.2f}\n")
                f.write(f"Maximum Drawdown: {report['Maximum Drawdown']:.2%}\n")
                f.write(f"Value at Risk (95%): {report['Value at Risk (95%)']:.2%}\n")
                f.write(f"Value at Risk (99%): {report['Value at Risk (99%)']:.2%}\n")
                f.write(f"Conditional VaR (95%): {report['Conditional VaR (95%)']:.2%}\n")
                f.write(f"Best Return: {report['Best Return']:.2%}\n")
                f.write(f"Worst Return: {report['Worst Return']:.2%}\n")
                f.write(f"Positive Returns Ratio: {report['Positive Returns Ratio']:.2%}\n\n")
                
                f.write("PORTFOLIO ALLOCATION:\n")
                for ticker, weight in zip(self.tickers, weights):
                    f.write(f"{ticker}: {weight:.2%}\n")
        
        return report
    
    def save_data(self, filepath):
        """
        Save data to CSV file
        
        Parameters:
        -----------
        filepath : str
            Path to save the data
        """
        self.data_manager.save_data(filepath)
    
    def load_data(self, filepath):
        """
        Load data from CSV file
        
        Parameters:
        -----------
        filepath : str
            Path to load the data from
            
        Returns:
        --------
        pd.DataFrame : Loaded data
        """
        self.data = self.data_manager.load_data(filepath)
        return self.data
