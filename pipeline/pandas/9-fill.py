#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Remove the Weighted_Price column if it exists
if 'Weighted_Price' in df.columns:
    df.drop(columns=['Weighted_Price'], inplace=True)

# Fill missing values in Close with the previous row's value
df['Close'].fillna(method='ffill', inplace=True)

# Fill missing values in High, Low, and Open with the same row's Close value
df['High'].fillna(df['Close'], inplace=True)
df['Low'].fillna(df['Close'], inplace=True)
df['Open'].fillna(df['Close'], inplace=True)

# Fill missing values in Volume_(BTC) and Volume_(Currency) with 0
df['Volume_(BTC)'].fillna(0, inplace=True)
df['Volume_(Currency)'].fillna(0, inplace=True)

print(df.head())
print(df.tail())
