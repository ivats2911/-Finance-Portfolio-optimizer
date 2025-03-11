"""
Basic usage example for the portfolio optimizer.

This example demonstrates how to use the PortfolioOptimizer class to:
1. Fetch historical data for a set of stocks
2. Calculate optimal portfolios
3. Visualize the efficient frontier
4. Generate a performance report
"""

import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Add the parent directory to the path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Direct imports from local modules
from optimizer import PortfolioOptimizer


def main():
    # Define portfolio constituents
    tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'BRK-B', 
               'JNJ', 'PG', 'XOM', 'JPM', 'BAC', 'V', 'MA', 'DIS', 
               'NFLX', 'TSLA', 'NVDA', 'HD', 'VZ', 'KO']
    
    # Initialize optimizer with 5 years of historical data
    optimizer = PortfolioOptimizer(tickers)
    
    # Fetch data and calculate returns
    optimizer.fetch_data()
    optimizer.calculate_returns(frequency='daily')
    
    # Find optimal portfolios using different strategies
    max_sharpe_portfolio = optimizer.optimal_portfolio('max_sharpe')
    min_vol_portfolio = optimizer.optimal_portfolio('min_volatility')
    max_div_portfolio = optimizer.optimal_portfolio('max_diversification')
    risk_parity_portfolio = optimizer.optimal_portfolio('risk_parity')
    
    # Print results
    print("\nMaximum Sharpe Ratio Portfolio:")
    print(f"Annual Return: {max_sharpe_portfolio['performance'][0]:.2%}")
    print(f"Annual Volatility: {max_sharpe_portfolio['performance'][1]:.2%}")
    print(f"Sharpe Ratio: {max_sharpe_portfolio['performance'][2]:.2f}")
    print("\nAllocation:")
    for ticker, allocation in max_sharpe_portfolio['allocation'].items():
        if allocation > 1:  # Only show allocations > 1%
            print(f"{ticker}: {allocation:.2f}%")
    
    # Visualize efficient frontier
    ef_fig = optimizer.visualize_efficient_frontier(
        show_assets=True,
        show_monte_carlo=True,
        monte_carlo_simulations=1000
    )
    ef_fig.savefig('efficient_frontier.png')
    
    # Visualize allocation
    alloc_fig = optimizer.visualize_allocation(
        max_sharpe_portfolio['weights'],
        title="Optimal Portfolio Allocation (Maximum Sharpe Ratio)"
    )
    alloc_fig.savefig('portfolio_allocation.png')
    
    # Simulate market scenarios
    scenarios, scenario_fig = optimizer.simulate_scenarios(
        max_sharpe_portfolio['weights'],
        initial_investment=1000000  # £1M
    )
    scenario_fig.savefig('market_scenarios.png')
    
    # Generate stress test
    stress_results, stress_fig, _ = optimizer.generate_stress_test(
        max_sharpe_portfolio['weights'],
        initial_investment=1000000
    )
    stress_fig.savefig('stress_test.png')
    
    # Generate performance report
    report = optimizer.generate_performance_report(
        max_sharpe_portfolio['weights'],
        file_path='portfolio_report.txt'
    )
    
    print("\nAnalysis complete! Files saved:")
    print("- efficient_frontier.png")
    print("- portfolio_allocation.png")
    print("- market_scenarios.png")
    print("- stress_test.png")
    print("- portfolio_report.txt")
    
    # Compare strategies
    strategies = {
        "Max Sharpe Ratio": max_sharpe_portfolio,
        "Min Volatility": min_vol_portfolio,
        "Max Diversification": max_div_portfolio,
        "Risk Parity": risk_parity_portfolio
    }
    
    # Create comparison table
    comparison = pd.DataFrame({
        s: {
            "Annual Return": p["performance"][0],
            "Annual Volatility": p["performance"][1],
            "Sharpe Ratio": p["performance"][2],
            "Number of Assets > 1%": sum(1 for w in p["weights"] if w > 0.01)
        } for s, p in strategies.items()
    }).T
    
    # Format the table
    comparison["Annual Return"] = comparison["Annual Return"].map("{:.2%}".format)
    comparison["Annual Volatility"] = comparison["Annual Volatility"].map("{:.2%}".format)
    comparison["Sharpe Ratio"] = comparison["Sharpe Ratio"].map("{:.2f}".format)
    comparison["Number of Assets > 1%"] = comparison["Number of Assets > 1%"].astype(int)
    
    print("\nStrategy Comparison:")
    print(comparison)
    
    # Keep plots open
    plt.show()


if __name__ == "__main__":
    main()
