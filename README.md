# Black-Scholes Option Pricing Model Calculator

An interactive web application for calculating and visualizing option prices using the Black-Scholes model. Built with Streamlit, this tool provides real-time option pricing, P&L calculations, and dynamic visualizations.

## Features

- **Real-time Option Pricing**: Calculate European Call and Put option prices instantly
- **Interactive Parameters**: Adjust all Black-Scholes model inputs through an intuitive sidebar
- **P&L Analysis**: Track potential profits and losses with purchase price inputs
- **Visual Analytics**: 
  - Dynamic heatmaps showing price sensitivity to stock price and volatility changes
  - Color-coded P&L visualization (green for profit, red for loss)
  - Current position marker on heatmaps
- **Put-Call Parity**: Automatic verification of put-call parity relationship
- **Educational Tools**: Helpful tooltips and explanations throughout the interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/BSMModel.git
cd BSMModel
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Use the sidebar to adjust parameters:
   - Stock Price (S)
   - Strike Price (K)
   - Time to Expiration (T)
   - Risk-free Rate (r)
   - Volatility (σ)
   - Option Purchase Prices

4. View results in real-time:
   - Current option prices
   - P&L calculations
   - Price sensitivity heatmaps
   - Put-call parity verification

## Mathematical Background

### Key Parameters
| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Stock Price | S | Current market price of the underlying stock |
| Strike Price | K | Price at which the option can be exercised |
| Time to Expiration | T | Time until option expiry (in years) |
| Risk-free Rate | r | Annual risk-free interest rate (as decimal) |
| Volatility | σ (sigma) | Annual volatility of the stock (as decimal) |

### Option Price Formulas

#### Call Option Price (C):
```
C = S × N(d₁) - K × e^(-rT) × N(d₂)
```

#### Put Option Price (P):
```
P = K × e^(-rT) × N(-d₂) - S × N(-d₁)
```

#### Where d₁ and d₂ are:
```
d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
```

### Formula Components Explained
- `N(x)`: Cumulative standard normal distribution function
- `e`: Mathematical constant (≈ 2.71828)
- `ln(x)`: Natural logarithm
- `√T`: Square root of time to expiration

### Example
For a stock with:
- Stock Price (S) = $100
- Strike Price (K) = $100
- Time (T) = 1 year
- Risk-free Rate (r) = 5% (0.05)
- Volatility (σ) = 20% (0.20)

The formula would calculate:
1. First, find d₁ and d₂
2. Then plug these values into the call or put formula
3. The result gives the theoretical fair price of the option

### Put-Call Parity
The relationship between call and put prices:
```
C - P = S - K × e^(-rT)
```
This means that for the same strike price and expiration, the difference between call and put prices should equal the difference between the current stock price and the discounted strike price.

## Requirements

- Python 3.8+
- Streamlit
- NumPy
- SciPy
- Pandas
- Seaborn
- Matplotlib

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Black-Scholes-Merton model (1973)
- Streamlit framework
- Scientific Python community
