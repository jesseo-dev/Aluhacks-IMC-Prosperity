import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the dataset
# Note: The delimiter in your file is a semicolon ';'
file_path = 'prices_round_0_day_-1.csv'
df = pd.read_csv(file_path, sep=';')

# 2. Separate the data by product
products = df['product'].unique()

# 3. Create the visualization
fig, axes = plt.subplots(nrows=len(products), ncols=1, figsize=(12, 10), sharex=True)

for i, product in enumerate(products):
    product_data = df[df['product'] == product]
    
    # Plot Mid Price
    axes[i].plot(product_data['timestamp'], product_data['mid_price'], 
                 label=f'{product} Mid Price', color='tab:blue', linewidth=1)
    
    # Optional: Plot Bid Price 1 and Ask Price 1 to see the spread
    axes[i].fill_between(product_data['timestamp'], 
                         product_data['bid_price_1'], 
                         product_data['ask_price_1'], 
                         color='tab:gray', alpha=0.3, label='Bid-Ask Spread')
    
    axes[i].set_title(f'Price Trend for {product}', fontsize=14)
    axes[i].set_ylabel('Price')
    axes[i].legend(loc='upper right')
    axes[i].grid(True, linestyle='--', alpha=0.7)

plt.xlabel('Timestamp')
plt.tight_layout()

# 4. Save or Show the chart
plt.savefig('market_price_charts.png')
plt.show()

print("Charts generated successfully.")