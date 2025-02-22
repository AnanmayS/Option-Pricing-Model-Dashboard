import streamlit as st
import numpy as np
from scipy.stats import norm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Black-Scholes functions
def black_scholes_call(S, K, T, r, sigma):
    """Calculate the European call option price."""
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    """Calculate the European put option price."""
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def create_heatmap_data(K, T, r, S_range, sigma_range, call_purchase, put_purchase):
    """Create heatmap data for call and put PnL."""
    # Reduced number of points for better readability of annotations
    S_values = np.linspace(S_range[0], S_range[1], 15)
    sigma_values = np.linspace(sigma_range[0], sigma_range[1], 10)
    
    S_mesh, sigma_mesh = np.meshgrid(S_values, sigma_values)
    call_prices = np.zeros_like(S_mesh)
    put_prices = np.zeros_like(S_mesh)
    call_pnl = np.zeros_like(S_mesh)
    put_pnl = np.zeros_like(S_mesh)
    
    for i in range(len(sigma_values)):
        for j in range(len(S_values)):
            call_prices[i, j] = black_scholes_call(S_mesh[i, j], K, T, r, sigma_mesh[i, j])
            put_prices[i, j] = black_scholes_put(S_mesh[i, j], K, T, r, sigma_mesh[i, j])
            call_pnl[i, j] = call_prices[i, j] - call_purchase
            put_pnl[i, j] = put_prices[i, j] - put_purchase
    
    return S_values, sigma_values, call_pnl, put_pnl

# Main Streamlit application
def main():
    st.set_page_config(page_title="Black-Scholes Option Pricing Model", layout="wide")
    
    # Sidebar for parameters
    st.sidebar.title("Model Parameters")
    st.sidebar.markdown("---")
    
    # Market Parameters in sidebar
    st.sidebar.subheader("Market Parameters")
    S = st.sidebar.number_input("Current Stock Price (S)", 
                       min_value=0.01, 
                       value=100.0, 
                       step=1.0,
                       help="The current price of the underlying stock")
    
    sigma = st.sidebar.number_input("Volatility (σ)", 
                          min_value=0.01, 
                          max_value=2.0, 
                          value=0.2, 
                          step=0.01,
                          help="Annual volatility of the stock price (between 1% and 200%)")
    
    r = st.sidebar.number_input("Risk-Free Interest Rate (r)", 
                       min_value=-0.1, 
                       max_value=0.5, 
                       value=0.05, 
                       step=0.001,
                       format="%.3f",
                       help="Annual risk-free interest rate (as a decimal)")

    # Option Parameters in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Option Parameters")
    K = st.sidebar.number_input("Strike Price (K)", 
                       min_value=0.01, 
                       value=100.0, 
                       step=1.0,
                       help="The strike price of the option")
    
    T = st.sidebar.number_input("Time to Expiration (T in years)", 
                       min_value=0.01, 
                       max_value=10.0, 
                       value=1.0, 
                       step=0.1,
                       help="Time until option expiration in years")

    # Purchase Price inputs in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Purchase Prices")
    call_purchase = st.sidebar.number_input("Call Option Purchase Price", 
                                  min_value=0.0, 
                                  value=0.0, 
                                  step=0.1,
                                  help="Enter the price at which you purchased the call option")
    
    put_purchase = st.sidebar.number_input("Put Option Purchase Price", 
                                min_value=0.0, 
                                value=0.0, 
                                step=0.1,
                                help="Enter the price at which you purchased the put option")

    # Parameters summary in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Parameters Summary")
    params_data = {
        'Parameter': ['Stock Price (S)', 'Strike Price (K)', 'Time to Expiration (T)', 
                     'Risk-Free Rate (r)', 'Volatility (σ)', 'Call Purchase', 'Put Purchase'],
        'Value': [f'${S:.2f}', f'${K:.2f}', f'{T:.2f} years', 
                 f'{r:.1%}', f'{sigma:.1%}', f'${call_purchase:.2f}', f'${put_purchase:.2f}']
    }
    df = pd.DataFrame(params_data)
    st.sidebar.table(df)

    # Main content area
    st.title("Black-Scholes Option Pricing Model")
    st.markdown("""
    This application calculates European option prices using the Black-Scholes model.
    Adjust the parameters in the sidebar to see how they affect both Call and Put option prices.
    """)

    # Calculate prices
    if T <= 0 or sigma <= 0:
        st.error("Time to expiration and volatility must be greater than 0.")
    else:
        call_price = black_scholes_call(S, K, T, r, sigma)
        put_price = black_scholes_put(S, K, T, r, sigma)
        
        # Display option prices and PnL in large format
        st.markdown("### Current Option Prices and P&L")
        price_cols = st.columns(2)
        
        with price_cols[0]:
            st.metric(label="Call Option Price", value=f"${call_price:.2f}")
            call_pnl = call_price - call_purchase
            st.metric(
                label="Call Option P&L",
                value=f"${call_pnl:.2f}",
                delta=f"{(call_pnl/call_purchase*100):.1f}%" if call_purchase > 0 else "N/A",
                delta_color="normal"
            )
            
        with price_cols[1]:
            st.metric(label="Put Option Price", value=f"${put_price:.2f}")
            put_pnl = put_price - put_purchase
            st.metric(
                label="Put Option P&L",
                value=f"${put_pnl:.2f}",
                delta=f"{(put_pnl/put_purchase*100):.1f}%" if put_purchase > 0 else "N/A",
                delta_color="normal"
            )

        # Add explanation of PnL
        st.markdown("""
        #### P&L Explanation
        - **P&L = Current Price - Purchase Price**
        - Positive P&L indicates a profit, negative indicates a loss
        - Percentage shows the return on investment
        - 'N/A' percentage is shown when no purchase price is entered
        """)
        
        # Put-Call Parity in a smaller section
        st.markdown("#### Put-Call Parity Verification")
        parity_left = call_price - put_price
        parity_right = S - K * np.exp(-r * T)
        parity_cols = st.columns(2)
        with parity_cols[0]:
            st.write(f"C - P = ${parity_left:.2f}")
            st.write(f"S - Ke^(-rT) = ${parity_right:.2f}")
        with parity_cols[1]:
            if abs(parity_left - parity_right) < 0.01:
                st.success("✓ Put-Call Parity holds")
            else:
                st.warning("⚠ Put-Call Parity deviation detected")
            
        # Heatmaps
        st.markdown("### Price Sensitivity Heatmaps")
        st.markdown("""
        These heatmaps show how option prices vary with different combinations of stock price and volatility.
        The current values are marked with a white dot. Use the sliders below to adjust the ranges.
        """)
        
        # Add sliders for heatmap ranges
        heatmap_cols = st.columns(2)
        
        with heatmap_cols[0]:
            st.markdown("##### Stock Price Range")
            stock_range = st.slider(
                "Stock Price Range (±$ from current price)",
                min_value=10,
                max_value=200,
                value=50,
                step=10,
                help="Range of stock prices to show in heatmap (centered around current price)"
            )
            
        with heatmap_cols[1]:
            st.markdown("##### Volatility Range")
            max_vol = st.slider(
                "Maximum Volatility",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1,
                format="%.1f",
                help="Maximum volatility to show in heatmap"
            )
        
        # Create heatmap data with user-defined ranges
        S_range = [max(0.1, S - stock_range), S + stock_range]
        sigma_range = [0.05, max_vol]
        
        S_values, sigma_values, call_pnl_map, put_pnl_map = create_heatmap_data(
            K, T, r, S_range, sigma_range, call_purchase, put_purchase)
        
        # Create heatmaps with larger figure size and more height for annotations
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Function to get sparse tick positions
        def get_sparse_ticks(values, num_ticks=6):
            indices = np.linspace(0, len(values)-1, num_ticks, dtype=int)
            return indices, values[indices]
        
        # Get sparse tick positions
        x_indices, x_values = get_sparse_ticks(S_values)
        y_indices, y_values = get_sparse_ticks(sigma_values, 5)
        
        # Custom colormap for PnL (red for negative, green for positive)
        def create_pnl_colormap():
            # Create a custom colormap that's red for negative values and green for positive
            colors = [(0.8, 0.2, 0.2), (1, 1, 1), (0.2, 0.8, 0.2)]  # red, white, green
            return LinearSegmentedColormap.from_list('pnl_colormap', colors)
        
        # Function to create annotated heatmap
        def create_annotated_heatmap(data, ax, title):
            # Find the maximum absolute value for symmetric color scaling
            abs_max = max(abs(data.min()), abs(data.max()))
            
            # Create heatmap with custom colormap
            sns.heatmap(data, 
                       ax=ax, 
                       cmap=create_pnl_colormap(),
                       center=0,  # Center the colormap at 0
                       vmin=-abs_max,  # Symmetric limits
                       vmax=abs_max,
                       cbar_kws={'label': f'{title} P&L ($)', 'format': '%.1f'},
                       xticklabels=False,
                       yticklabels=False,
                       annot=True,
                       fmt='.1f',
                       annot_kws={'size': 8})
            
            # Manually set tick positions and labels
            ax.set_xticks(x_indices)
            ax.set_xticklabels([f'${x:.0f}' for x in x_values], rotation=45)
            ax.set_yticks(y_indices)
            ax.set_yticklabels([f'{y:.2f}' for y in y_values])
            
            ax.set_title(f'{title} P&L', pad=20, fontsize=12)
            ax.set_xlabel('Stock Price ($)', labelpad=10)
            ax.set_ylabel('Volatility (σ)', labelpad=10)
            
            # Mark current point
            S_idx = np.abs(S_values - S).argmin()
            sigma_idx = np.abs(sigma_values - sigma).argmin()
            ax.plot(S_idx, sigma_idx, 'wo', markersize=12, markeredgecolor='black')
        
        # Create both heatmaps with annotations
        create_annotated_heatmap(call_pnl_map, ax1, 'Call Option')
        create_annotated_heatmap(put_pnl_map, ax2, 'Put Option')
        
        # Add explanation of heatmap colors
        st.markdown("""
        #### Heatmap Color Guide
        - **Green**: Positive P&L (profit)
        - **Red**: Negative P&L (loss)
        - **White**: Break-even point (P&L ≈ 0)
        - Numbers show exact P&L value at each point
        - White dot marks current stock price and volatility
        """)
        
        # Adjust layout with more space for annotations
        plt.tight_layout()
        st.pyplot(fig)

if __name__ == '__main__':
    main()
