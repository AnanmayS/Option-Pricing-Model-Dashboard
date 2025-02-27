import streamlit as st
import numpy as np
from scipy.stats import norm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
from matplotlib.patches import Patch

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

# Binomial Tree functions
def binomial_tree_american(S, K, T, r, sigma, N, option_type='call'):
    """
    Price an American option using the binomial tree model.
    
    Parameters:
    S: Current stock price
    K: Strike price
    T: Time to expiration (in years)
    r: Risk-free interest rate
    sigma: Volatility
    N: Number of steps in the binomial tree
    option_type: 'call' or 'put'
    
    Returns:
    option_price: Price of the option
    stock_tree: Tree of stock prices
    option_tree: Tree of option values
    exercise_boundary: Nodes where early exercise is optimal
    """
    # Time step
    dt = T / N
    
    # Up and down factors
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    
    # Risk-neutral probability
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Initialize stock price tree
    stock_tree = np.zeros((N + 1, N + 1))
    
    # Initialize option value tree
    option_tree = np.zeros((N + 1, N + 1))
    
    # Initialize exercise boundary tracker
    exercise_boundary = np.zeros((N + 1, N + 1), dtype=bool)
    
    # Build stock price tree
    for i in range(N + 1):
        for j in range(i + 1):
            # Stock price at node (i, j)
            stock_tree[i, j] = S * (u ** (i - j)) * (d ** j)
    
    # Option payoffs at expiration
    for j in range(N + 1):
        if option_type.lower() == 'call':
            option_tree[N, j] = max(0, stock_tree[N, j] - K)
        else:  # put
            option_tree[N, j] = max(0, K - stock_tree[N, j])
    
    # Backward induction through the tree
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            # Expected option value (discounted)
            expected = np.exp(-r * dt) * (p * option_tree[i + 1, j] + (1 - p) * option_tree[i + 1, j + 1])
            
            # Immediate exercise value
            if option_type.lower() == 'call':
                exercise = max(0, stock_tree[i, j] - K)
            else:  # put
                exercise = max(0, K - stock_tree[i, j])
            
            # Option value is maximum of expected value and exercise value
            option_tree[i, j] = max(expected, exercise)
            
            # Track if early exercise is optimal
            if exercise > expected and exercise > 0:
                exercise_boundary[i, j] = True
    
    return option_tree[0, 0], stock_tree, option_tree, exercise_boundary

def plot_binomial_tree(stock_tree, option_tree, exercise_boundary, N_display, option_type):
    """
    Plot a visual representation of the binomial tree.
    
    Parameters:
    stock_tree: Tree of stock prices
    option_tree: Tree of option values
    exercise_boundary: Nodes where early exercise is optimal
    N_display: Number of steps to display (to avoid overcrowding)
    option_type: 'call' or 'put'
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Limit the display to N_display steps
    N_actual = min(N_display, stock_tree.shape[0] - 1)
    
    # Add nodes and edges
    pos = {}
    node_colors = []
    node_sizes = []
    labels = {}
    
    for i in range(N_actual + 1):
        for j in range(i + 1):
            # Node ID
            node_id = f"{i}_{j}"
            
            # Node position
            pos[node_id] = (i, N_actual - i + 2*j)
            
            # Node color based on exercise boundary
            if exercise_boundary[i, j]:
                node_colors.append('red')
            else:
                node_colors.append('skyblue')
            
            # Node size
            node_sizes.append(700)
            
            # Node label
            stock_price = stock_tree[i, j]
            option_value = option_tree[i, j]
            labels[node_id] = f"S: ${stock_price:.2f}\nO: ${option_value:.2f}"
            
            # Add node
            G.add_node(node_id)
            
            # Add edges
            if i < N_actual:
                G.add_edge(node_id, f"{i+1}_{j}")
                G.add_edge(node_id, f"{i+1}_{j+1}")
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Draw the graph
    nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=node_sizes, 
            arrows=True, arrowsize=15, edge_color='gray')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    
    # Add legend
    legend_elements = [
        Patch(facecolor='skyblue', edgecolor='black', label='Hold Option'),
        Patch(facecolor='red', edgecolor='black', label='Exercise Option')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Add title
    plt.title(f"Binomial Tree for American {option_type.capitalize()} Option (First {N_actual+1} steps)")
    
    # Remove axes
    plt.axis('off')
    
    return plt.gcf()

# Helper function to calculate intrinsic value
def calculate_intrinsic_value(S, K, option_type):
    """Calculate the intrinsic value of an option."""
    if option_type.lower() == 'call':
        return max(0, S - K)
    else:  # put
        return max(0, K - S)

# Helper function to suggest price range
def suggest_price_range(theoretical_price, intrinsic_value):
    """Suggest a realistic price range for an option."""
    min_price = intrinsic_value
    
    # Add a small spread around theoretical price (simulating bid-ask)
    lower_bound = max(min_price, theoretical_price * 0.95)
    upper_bound = theoretical_price * 1.05
    
    return min_price, lower_bound, theoretical_price, upper_bound

# Black-Scholes Model Page
def black_scholes_page():
    st.title("Black-Scholes Option Pricing Model")
    st.markdown("""
    This application calculates European option prices using the Black-Scholes model.
    Adjust the parameters in the sidebar to see how they affect both Call and Put option prices.
    """)
    
    # Sidebar for parameters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Parameters")
    
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

    # Calculate theoretical prices first to use in purchase price guidance
    if T <= 0 or sigma <= 0:
        st.error("Time to expiration and volatility must be greater than 0.")
    else:
        call_price = black_scholes_call(S, K, T, r, sigma)
        put_price = black_scholes_put(S, K, T, r, sigma)
        
        # Calculate intrinsic values
        call_intrinsic = calculate_intrinsic_value(S, K, 'call')
        put_intrinsic = calculate_intrinsic_value(S, K, 'put')
        
        # Suggest price ranges
        call_min, call_lower, call_theoretical, call_upper = suggest_price_range(call_price, call_intrinsic)
        put_min, put_lower, put_theoretical, put_upper = suggest_price_range(put_price, put_intrinsic)
        
        # Purchase Price inputs in sidebar with guidance
        st.sidebar.markdown("---")
        st.sidebar.subheader("Purchase Prices")
        
        # Call purchase price with guidance
        st.sidebar.markdown(f"""
        **Call Option Price Guidance:**
        - Intrinsic Value: ${call_intrinsic:.2f}
        - Theoretical Price: ${call_price:.2f}
        - Suggested Range: ${call_lower:.2f} - ${call_upper:.2f}
        """)
        
        call_purchase = st.sidebar.number_input("Call Option Purchase Price", 
                                min_value=0.0, 
                                value=0.0, 
                                step=0.1,
                                help="Enter the price at which you purchased the call option")
        
        # Warning if below intrinsic value
        if call_purchase > 0 and call_purchase < call_intrinsic:
            st.sidebar.warning(f"⚠️ Call purchase price (${call_purchase:.2f}) is below intrinsic value (${call_intrinsic:.2f}). This is unrealistic in actual markets.")
        
        # Put purchase price with guidance
        st.sidebar.markdown(f"""
        **Put Option Price Guidance:**
        - Intrinsic Value: ${put_intrinsic:.2f}
        - Theoretical Price: ${put_price:.2f}
        - Suggested Range: ${put_lower:.2f} - ${put_upper:.2f}
        """)
        
        put_purchase = st.sidebar.number_input("Put Option Purchase Price", 
                                min_value=0.0, 
                                value=0.0, 
                                step=0.1,
                                help="Enter the price at which you purchased the put option")
        
        # Warning if below intrinsic value
        if put_purchase > 0 and put_purchase < put_intrinsic:
            st.sidebar.warning(f"⚠️ Put purchase price (${put_purchase:.2f}) is below intrinsic value (${put_intrinsic:.2f}). This is unrealistic in actual markets.")

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

# Binomial Tree Model Page
def binomial_tree_page():
    st.title("American Option Pricing - Binomial Tree Model")
    st.markdown("""
    This model prices American options using a binomial tree approach. American options can be exercised at any time 
    before expiration, making them more complex to price than European options.
    
    The binomial tree model:
    - Divides time to expiration into discrete steps
    - Models possible stock price paths
    - Allows for evaluation of early exercise at each node
    - Converges to the Black-Scholes price for European options as steps increase
    """)
    
    # Sidebar for parameters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Parameters")
    
    # Market Parameters in sidebar
    st.sidebar.subheader("Market Parameters")
    S = st.sidebar.number_input("Current Stock Price (S)", 
                       min_value=0.01, 
                       value=100.0, 
                       step=1.0,
                       help="The current price of the underlying stock",
                       key="bt_S")
    
    sigma = st.sidebar.number_input("Volatility (σ)", 
                          min_value=0.01, 
                          max_value=2.0, 
                          value=0.2, 
                          step=0.01,
                          help="Annual volatility of the stock price (between 1% and 200%)",
                          key="bt_sigma")
    
    r = st.sidebar.number_input("Risk-Free Interest Rate (r)", 
                       min_value=-0.1, 
                       max_value=0.5, 
                       value=0.05, 
                       step=0.001,
                       format="%.3f",
                       help="Annual risk-free interest rate (as a decimal)",
                       key="bt_r")

    # Option Parameters in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Option Parameters")
    K = st.sidebar.number_input("Strike Price (K)", 
                       min_value=0.01, 
                       value=100.0, 
                       step=1.0,
                       help="The strike price of the option",
                       key="bt_K")
    
    T = st.sidebar.number_input("Time to Expiration (T in years)", 
                       min_value=0.01, 
                       max_value=10.0, 
                       value=1.0, 
                       step=0.1,
                       help="Time until option expiration in years",
                       key="bt_T")
    
    option_type = st.sidebar.selectbox(
        "Option Type",
        ["Call", "Put"],
        help="Call option gives the right to buy, Put option gives the right to sell"
    )
    
    N = st.sidebar.slider(
        "Number of Steps (N)",
        min_value=5,
        max_value=100,
        value=20,
        step=5,
        help="Number of time steps in the binomial tree. Higher values give more accurate prices but take longer to compute."
    )
    
    N_display = st.sidebar.slider(
        "Steps to Display",
        min_value=2,
        max_value=10,
        value=5,
        step=1,
        help="Number of steps to show in the tree visualization (limited for clarity)"
    )
    
    # Calculate option price
    if T <= 0 or sigma <= 0:
        st.error("Time to expiration and volatility must be greater than 0.")
    else:
        # Calculate option price using binomial tree
        option_price, stock_tree, option_tree, exercise_boundary = binomial_tree_american(
            S, K, T, r, sigma, N, option_type.lower())
        
        # Calculate intrinsic value
        intrinsic_value = calculate_intrinsic_value(S, K, option_type.lower())
        
        # Suggest price range
        min_price, lower_bound, theoretical_price, upper_bound = suggest_price_range(option_price, intrinsic_value)
        
        # Purchase Price input in sidebar with guidance
        st.sidebar.markdown("---")
        st.sidebar.subheader("Purchase Price")
        
        st.sidebar.markdown(f"""
        **{option_type} Option Price Guidance:**
        - Intrinsic Value: ${intrinsic_value:.2f}
        - Theoretical Price: ${option_price:.2f}
        - Suggested Range: ${lower_bound:.2f} - ${upper_bound:.2f}
        """)
        
        purchase_price = st.sidebar.number_input(
            f"{option_type} Option Purchase Price", 
            min_value=0.0, 
            value=0.0, 
            step=0.1,
            help=f"Enter the price at which you purchased the {option_type.lower()} option",
            key="bt_purchase"
        )
        
        # Warning if below intrinsic value
        if purchase_price > 0 and purchase_price < intrinsic_value:
            st.sidebar.warning(f"⚠️ Purchase price (${purchase_price:.2f}) is below intrinsic value (${intrinsic_value:.2f}). This is unrealistic in actual markets.")
        
        # Parameters summary in sidebar
        st.sidebar.markdown("---")
        st.sidebar.subheader("Parameters Summary")
        params_data = {
            'Parameter': ['Stock Price (S)', 'Strike Price (K)', 'Time to Expiration (T)', 
                        'Risk-Free Rate (r)', 'Volatility (σ)', f'{option_type} Purchase', 'Steps (N)'],
            'Value': [f'${S:.2f}', f'${K:.2f}', f'{T:.2f} years', 
                    f'{r:.1%}', f'{sigma:.1%}', f'${purchase_price:.2f}', f'{N}']
        }
        df = pd.DataFrame(params_data)
        st.sidebar.table(df)
        
        # Display option price and PnL
        st.markdown("### American Option Price and P&L")
        
        # Calculate P&L
        pnl = option_price - purchase_price
        
        # Display metrics
        st.metric(
            label=f"American {option_type} Option Price", 
            value=f"${option_price:.2f}"
        )
        
        st.metric(
            label=f"{option_type} Option P&L",
            value=f"${pnl:.2f}",
            delta=f"{(pnl/purchase_price*100):.1f}%" if purchase_price > 0 else "N/A",
            delta_color="normal"
        )
        
        # Compare with European option
        if option_type.lower() == 'call':
            european_price = black_scholes_call(S, K, T, r, sigma)
        else:
            european_price = black_scholes_put(S, K, T, r, sigma)
        
        # Calculate early exercise premium
        early_exercise_premium = option_price - european_price
        
        st.markdown("### American vs. European Option")
        cols = st.columns(3)
        
        with cols[0]:
            st.metric(
                label=f"American {option_type} Price", 
                value=f"${option_price:.2f}"
            )
        
        with cols[1]:
            st.metric(
                label=f"European {option_type} Price", 
                value=f"${european_price:.2f}"
            )
        
        with cols[2]:
            st.metric(
                label="Early Exercise Premium", 
                value=f"${early_exercise_premium:.2f}",
                delta=f"{(early_exercise_premium/european_price*100):.2f}%" if european_price > 0 else "N/A"
            )
        
        # Explanation of early exercise premium
        st.markdown("""
        #### Early Exercise Premium
        The early exercise premium is the additional value of an American option over its European counterpart.
        This premium represents the value of the right to exercise the option before expiration.
        
        - **Positive premium**: Early exercise may be optimal in some scenarios
        - **Zero premium**: Early exercise is never optimal (American = European)
        """)
        
        # Binomial Tree Visualization
        st.markdown("### Binomial Tree Visualization")
        st.markdown("""
        This visualization shows the first few steps of the binomial tree model:
        - Each node shows the stock price (S) and option value (O)
        - **Red nodes** indicate where early exercise is optimal
        - **Blue nodes** indicate where holding the option is optimal
        """)
        
        # Plot the tree
        fig = plot_binomial_tree(stock_tree, option_tree, exercise_boundary, N_display, option_type.lower())
        st.pyplot(fig)
        
        # Add note about visualization limitations
        st.info(f"""
        Note: For clarity, only the first {N_display} steps of the {N}-step tree are shown in the visualization.
        The option price calculation uses the full {N}-step tree.
        """)
        
        # Convergence analysis
        st.markdown("### Convergence Analysis")
        st.markdown("""
        The binomial tree model converges to the true option price as the number of steps increases.
        This chart shows how the option price changes with different numbers of steps.
        """)
        
        # Calculate prices for different numbers of steps
        step_values = list(range(5, min(N+5, 50), 5))
        prices = []
        
        for n_steps in step_values:
            price, _, _, _ = binomial_tree_american(S, K, T, r, sigma, n_steps, option_type.lower())
            prices.append(price)
        
        # Create convergence plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(step_values, prices, marker='o', linestyle='-', color='blue')
        ax.axhline(y=option_price, color='red', linestyle='--', alpha=0.7, label=f'Current ({N} steps)')
        
        if option_type.lower() == 'call':
            ax.axhline(y=european_price, color='green', linestyle=':', alpha=0.7, label='European (BS)')
        else:
            ax.axhline(y=european_price, color='green', linestyle=':', alpha=0.7, label='European (BS)')
        
        ax.set_xlabel('Number of Steps')
        ax.set_ylabel('Option Price ($)')
        ax.set_title(f'Convergence of American {option_type} Option Price')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        st.pyplot(fig)

# Main Streamlit application
def main():
    st.set_page_config(page_title="Option Pricing Models", layout="wide")
    
    # Create navigation in the sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a Model:",
        ["Black-Scholes Model", "Binomial Tree Model"]
    )
    
    # Display the selected page
    if page == "Black-Scholes Model":
        black_scholes_page()
    elif page == "Binomial Tree Model":
        binomial_tree_page()

if __name__ == '__main__':
    main()
