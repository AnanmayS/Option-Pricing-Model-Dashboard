# Option Pricing Models Calculator

An interactive web application for calculating and visualizing option prices using multiple pricing models. Built with Streamlit, this tool provides real-time option pricing, P&L calculations, and dynamic visualizations.

## Features

### Black-Scholes Model (European Options)
- **Real-time Option Pricing**: Calculate European Call and Put option prices instantly
- **Interactive Parameters**: Adjust all Black-Scholes model inputs through an intuitive sidebar
- **P&L Analysis**: Track potential profits and losses with purchase price inputs
- **Visual Analytics**: 
  - Dynamic heatmaps showing price sensitivity to stock price and volatility changes
  - Color-coded P&L visualization (green for profit, red for loss)
  - Current position marker on heatmaps
- **Put-Call Parity**: Automatic verification of put-call parity relationship

### Binomial Tree Model (American Options)
- **American Option Pricing**: Calculate prices for options that can be exercised before expiration
- **Early Exercise Analysis**: Visualize where early exercise is optimal in the option's lifetime
- **Tree Visualization**: Interactive binomial tree showing stock prices and option values at each node
- **Comparative Analysis**: Direct comparison between American and European option prices
- **Convergence Analysis**: Visualization of how option prices converge as the number of steps increases
- **Early Exercise Premium**: Calculation of the additional value from early exercise rights

### General Features
- **Multi-model Navigation**: Easily switch between different pricing models
- **Educational Tools**: Helpful tooltips and explanations throughout the interface
- **Realistic Price Guidance**: Suggested price ranges and intrinsic value warnings for realistic option pricing
- **Responsive Design**: Optimized for both desktop and mobile viewing

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AnanmayS/BSMModel.git
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Use the navigation in the sidebar to select a pricing model:
   - Black-Scholes Model (for European options)
   - Binomial Tree Model (for American options)

4. Adjust the parameters in the sidebar for your selected model:
   - Stock Price (S)
   - Strike Price (K)
   - Time to Expiration (T)
   - Risk-free Rate (r)
   - Volatility (σ)
   - Option Purchase Prices (with realistic price guidance)
   - For Binomial Tree: Number of steps (N)

5. View results in real-time:
   - Current option prices
   - P&L calculations
   - Model-specific visualizations

## Price Guidance Feature

The application provides realistic price guidance for options:

- **Intrinsic Value**: The minimum theoretical value of an option (what you'd get if exercised immediately)
  - For calls: max(0, Stock Price - Strike Price)
  - For puts: max(0, Strike Price - Stock Price)

- **Theoretical Price**: The model-calculated fair value of the option

- **Suggested Price Range**: A realistic range around the theoretical price (simulating market bid-ask spread)

- **Unrealistic Price Warning**: Visual warning when a purchase price is set below intrinsic value

This feature helps users understand realistic option pricing while still allowing for hypothetical scenarios.

## Mathematical Background

### Black-Scholes Model (European Options)

#### Key Parameters
| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Stock Price | S | Current market price of the underlying stock |
| Strike Price | K | Price at which the option can be exercised |
| Time to Expiration | T | Time until option expiry (in years) |
| Risk-free Rate | r | Annual risk-free interest rate (as decimal) |
| Volatility | σ (sigma) | Annual volatility of the stock (as decimal) |

#### Option Price Formulas

##### Call Option Price (C):
```
C = S × N(d₁) - K × e^(-rT) × N(d₂)
```

##### Put Option Price (P):
```
P = K × e^(-rT) × N(-d₂) - S × N(-d₁)
```

##### Where d₁ and d₂ are:
```
d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
```

#### Put-Call Parity
The relationship between call and put prices:
```
C - P = S - K × e^(-rT)
```

### Binomial Tree Model (American Options)

#### Key Concepts
- **Discrete Time Steps**: The time to expiration is divided into N discrete periods
- **Stock Price Movement**: At each step, the stock can move up or down by factors u and d
- **Risk-Neutral Probability**: The probability of an up move in a risk-neutral world
- **Backward Induction**: Option values are calculated from expiration backward to the present
- **Early Exercise**: At each node, the option value is the maximum of continuation value and immediate exercise value

#### Model Parameters
- All Black-Scholes parameters, plus:
- **Number of Steps (N)**: More steps provide higher accuracy but require more computation

#### Key Formulas
```
u = e^(σ√(T/N))  # Up factor
d = 1/u          # Down factor
p = (e^(rT/N) - d) / (u - d)  # Risk-neutral probability
```

#### Early Exercise Premium
The additional value of an American option over its European counterpart:
```
Premium = American Option Price - European Option Price
```

## Requirements

- Python 3.8+
- Streamlit
- NumPy
- SciPy
- Pandas
- Seaborn
- Matplotlib
- NetworkX (for tree visualization)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Black-Scholes-Merton model (1973)
- Cox-Ross-Rubinstein binomial model (1979)
- Streamlit framework
- Scientific Python community
