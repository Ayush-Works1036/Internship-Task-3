import pandas as pd
import numpy as np

np.random.seed(42)
dates = pd.date_range("2023-01-01", "2025-12-31", freq="D")
n = len(dates)
t = np.arange(n)
sales = (120 + np.linspace(0, 70, n)
         + 15*np.sin(2*np.pi*t/365.25)
         + 8*np.sin(2*np.pi*t/7)
         + np.random.normal(0, 10, n))
sales = np.maximum(sales, 20).round(2)

pd.DataFrame({"Date": dates, "Sales": sales}).to_csv("historical_sales.csv", index=False)
print("historical_sales.csv created successfully.")
