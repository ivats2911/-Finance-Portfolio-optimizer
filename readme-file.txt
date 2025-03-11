# Portfolio Optimizer

A comprehensive framework for portfolio optimization, risk analysis, and performance simulation.

## Features

- **Data Management**: Fetch and process historical financial data
- **Optimization Strategies**: Implement various portfolio optimization approaches
  - Maximum Sharpe ratio
  - Minimum volatility
  - Maximum diversification
  - Risk parity
- **Risk Analysis**: Calculate key risk and performance metrics
  - Sharpe ratio, Sortino ratio, Calmar ratio
  - Maximum drawdown
  - Value at Risk (VaR) and Conditional VaR
- **Market Simulations**: Project portfolio performance under different scenarios
  - Bull/bear markets
  - High/low volatility environments
  - Stress testing
- **Visualization Tools**: Create insightful charts and reports
  - Efficient frontier
  - Asset allocation
  - Performance projections

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/portfolio-optimizer.git
cd portfolio-optimizer

# Install required packages
pip install numpy pandas matplotlib seaborn scipy yfinance
```

## Usage

### Basic Example

```python
from portfolio_optimizer import PortfolioOptimizer

# Define portfolio constituents
tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'FB']

# Initialize optimizer
optimizer = PortfolioOptimizer(tickers)

# Fetch data and calculate returns
optimizer.fetch_data()
optimizer.calculate_returns()

# Find optimal portfolio
max_sharpe_portfolio = optimizer.optimal_portfolio('max_sharpe')

# Print results
print(f"Annual Return: {max_sharpe_portfolio['performance'][0]:.2%}")
print(f"Annual Volatility: {max_sharpe_portfolio['performance'][1]:.2%}")
print(f"Sharpe Ratio: {max_sharpe_portfolio['performance'][2]:.2f}")

# Visualize efficient frontier
optimizer.visualize_efficient_frontier(show_assets=True)
plt.show()
```

### Advanced Usage

For more advanced usage examples, see the scripts in the `examples/` directory:

- `basic_usage.py`: Basic portfolio optimization
- `efficient_frontier.py`: Detailed analysis of the efficient frontier
- `market_scenarios.py`: Simulating portfolio performance under different market conditions

## Project Structure

```
portfolio_optimizer/
│
├── __init__.py                 # Makes portfolio_optimizer a Python package
├── optimizer.py                # Main optimizer class
├── data_manager.py             # Data fetching and processing
├── strategies.py               # Optimization strategies
├── risk_metrics.py             # Risk and performance metrics calculations
├── visualizations.py           # Plotting and visualization functions
├── market_simulator.py         # Market scenario simulation
│
├── examples/                   # Example scripts
│   ├── __init__.py
│   ├── basic_usage.py          # Basic usage example
│   ├── efficient_frontier.py   # Efficient frontier example
│   └── market_scenarios.py     # Market simulation example
```

## Dependencies

- NumPy: Numerical computations
- Pandas: Data manipulation
- Matplotlib/Seaborn: Visualization
- SciPy: Optimization algorithms
- yfinance: Yahoo Finance data API

## License

This project is licensed under the MIT License - see the LICENSE file for details.
