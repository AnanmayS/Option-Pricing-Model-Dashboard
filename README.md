# Black-Scholes Option Pricing Model Dashboard

A Streamlit web application for calculating and visualizing European option prices using the Black-Scholes model. This interactive dashboard allows users to explore how different parameters affect option prices and profit/loss (P&L) calculations.

## Features

- Real-time calculation of European Call and Put option prices
- Interactive parameter adjustment through sidebar controls
- Visual P&L analysis through dynamic heatmaps
- Put-Call parity verification
- Comprehensive parameter summary display
- P&L calculations with purchase price tracking
- Responsive visualization of price sensitivities

## Requirements

```
streamlit
numpy
scipy
pandas
seaborn
matplotlib
```

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Access the dashboard through your web browser (typically at `http://localhost:8501`)

3. Adjust parameters using the sidebar controls:
   - Market Parameters:
     - Current Stock Price (S)
     - Volatility (σ)
     - Risk-Free Interest Rate (r)
   - Option Parameters:
     - Strike Price (K)
     - Time to Expiration (T)
   - Purchase Prices:
     - Call Option Purchase Price
     - Put Option Purchase Price

## Features Explanation

### Main Dashboard
- **Current Option Prices**: Real-time display of calculated Call and Put option prices
- **P&L Metrics**: Shows current profit/loss based on purchase prices
- **Put-Call Parity**: Verification of the put-call parity relationship
- **Price Sensitivity Heatmaps**: Visual representation of how option prices change with stock price and volatility

### Interactive Controls
- Adjustable stock price range for heatmaps
- Configurable maximum volatility display
- Comprehensive parameter summary in sidebar

### Visualization
- Color-coded heatmaps (green for profit, red for loss)
- Current price/volatility point marked with white dot
- Numerical P&L values displayed in heatmap cells

## Technical Details

The application implements the Black-Scholes model using the following key functions:

- `black_scholes_call()`: Calculates European call option prices
- `black_scholes_put()`: Calculates European put option prices
- `create_heatmap_data()`: Generates data for P&L visualization

## Limitations

- Only supports European-style options
- Assumes constant volatility and risk-free rate
- Does not account for dividends
- Simplified market assumptions inherent to the Black-Scholes model

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

[MIT License](LICENSE)
