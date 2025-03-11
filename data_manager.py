"""
Data management module for the portfolio optimizer.

This module contains functions for fetching, processing, and managing
financial data for portfolio optimization.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta


class DataManager:
    """
    Class for fetching and managing financial data for portfolio optimization.
    """
    
    def __init__(self, tickers, start_date=None, end_date=None):
        """
        Initialize the data manager with stock tickers and date range.
        
        Parameters:
        -----------
        tickers : list
            List of stock ticker symbols
        start_date : str, optional
            Start date for historical data in 'YYYY-MM-DD' format
        end_date : str, optional
            End date for historical data in 'YYYY-MM-DD' format
        """
        self.tickers = tickers
        
        # Default to 5 years of data if dates not specified
        if end_date is None:
            self.end_date = datetime.now()
        else:
            self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
            
        if start_date is None:
            self.start_date = self.end_date - timedelta(days=5*365)
        else:
            self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
            
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
        print(f"Fetching data for {len(self.tickers)} securities...")
        
        # Download data - note that auto_adjust=True is now the default
        # so we'll get adjusted close prices directly
        data = yf.download(
            self.tickers,
            start=self.start_date.strftime('%Y-%m-%d'),
            end=self.end_date.strftime('%Y-%m-%d'),
            progress=False
        )
        
        # Check the columns to decide how to access the data
        if isinstance(data.columns, pd.MultiIndex):
            # If we have a MultiIndex, get the 'Close' prices
            self.data = data['Close']
        else:
            # If we have a single ticker, columns won't be MultiIndex
            self.data = data['Close']
            # Handle case where only one ticker is provided
            if len(self.tickers) == 1:
                self.data = pd.DataFrame(self.data, columns=self.tickers)
        
        print(f"Data collected from {self.data.index[0]} to {self.data.index[-1]}")
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
        tuple : (returns, mean_returns, cov_matrix)
        """
        if self.data is None:
            self.fetch_data()
            
        if frequency == 'daily':
            self.returns = self.data.pct_change().dropna()
        elif frequency == 'monthly':
            monthly_data = self.data.resample('M').last()
            self.returns = monthly_data.pct_change().dropna()
        elif frequency == 'annual':
            annual_data = self.data.resample('Y').last()
            self.returns = annual_data.pct_change().dropna()
        
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()
        
        if frequency == 'daily':
            # Annualize for daily data
            self.mean_returns = self.mean_returns * 252
            self.cov_matrix = self.cov_matrix * 252
            
        elif frequency == 'monthly':
            # Annualize for monthly data
            self.mean_returns = self.mean_returns * 12
            self.cov_matrix = self.cov_matrix * 12
        
        return self.returns, self.mean_returns, self.cov_matrix
    
    def get_risk_free_rate(self, source='default'):
        """
        Get risk-free rate from various sources
        
        Parameters:
        -----------
        source : str
            Source for risk-free rate ('default', 'treasury', etc.)
            
        Returns:
        --------
        float : Risk-free rate as decimal
        """
        # For now, return a default value
        # This could be extended to fetch from treasury websites or other sources
        return 0.01  # 1% risk-free rate
    
    def save_data(self, filepath):
        """
        Save data to CSV file
        
        Parameters:
        -----------
        filepath : str
            Path to save the data
        """
        if self.data is not None:
            self.data.to_csv(filepath)
    
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
        self.data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        return self.data
