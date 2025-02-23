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
git clone https://github.com/AnanmayS/Black-Scholes-Option-Pricing-Model-Dashboard.git
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

The Black-Scholes model uses the following formulas:

Call Option Price:
\[ C = SN(d_1) - Ke^{-rT}N(d_2) \]

Put Option Price:
\[ P = Ke^{-rT}N(-d_2) - SN(-d_1) \]

Where:
\[ d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}} \]
\[ d_2 = d_1 - \sigma\sqrt{T} \]

- S: Current stock price
- K: Strike price
- T: Time to expiration (in years)
- r: Risk-free interest rate
- σ: Volatility
- N(): Cumulative standard normal distribution function

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
