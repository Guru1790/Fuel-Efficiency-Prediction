# -*- coding: utf-8 -*-
"""Fuel Efficiency prediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1RuBiBNu1EaMPdF_tFj97MJVQNpOcG_5i
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

import tensorflow as tf
from tensorflow import keras
from keras import layers

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/content/auto-mpg.csv')
df.head()

df.shape

df.info()

df.describe()

"""Exploratory Data Analysis"""

df['horsepower'].unique()

print(df.shape)
df = df[df['horsepower'] != '?']
print(df.shape)

df['horsepower'] = df['horsepower'].astype(int)
df.isnull().sum()

df['horsepower'].replace('?', np.nan, inplace=True)
df.dropna(subset=['horsepower'], inplace=True)
print(df.shape)

# Convert 'horsepower' to numeric
df['horsepower'] = df['horsepower'].astype(float)

df.nunique()

# Display data types of each column
print(df.dtypes)

# Inspect columns with object data type
for col in df.select_dtypes(include=['object']).columns:
    print(f"Unique values in {col}:")
    print(df[col].unique())
    print("\n")

# Identify and inspect non-numeric columns
for col in df.select_dtypes(include=['object']).columns:
    print(f"Unique values in {col}:")
    print(df[col].unique())
    print("\n")

df = df.drop(columns=['car name'])

df['horsepower'].replace('?', np.nan, inplace=True)
df.dropna(subset=['horsepower'], inplace=True)
df['horsepower'] = df['horsepower'].astype(float)

plt.subplots(figsize=(15, 5))
for i, col in enumerate(['cylinders', 'origin']):
    plt.subplot(1, 2, i+1)
    x = df.groupby(col).mean()['mpg']
    x.plot.bar()
    plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 8))
sb.heatmap(df.corr() > 0.9,
		annot=True,
		cbar=False)
plt.show()

df.drop('displacement',
		axis=1,
		inplace=True)

# Check the columns in the DataFrame
print(df.columns)

# Correct the drop command if necessary
features = df.drop(['mpg', 'car name'], axis=1, errors='ignore')  # 'errors' to ignore if not found
target = df['mpg'].values

# Proceed with the train-test split
X_train, X_val, Y_train, Y_val = train_test_split(
    features, target, test_size=0.2, random_state=22
)

# Check the shapes of the training and validation sets
X_train.shape, X_val.shape

AUTO = tf.data.experimental.AUTOTUNE

train_ds = (
	tf.data.Dataset
	.from_tensor_slices((X_train, Y_train))
	.batch(32)
	.prefetch(AUTO)
)

val_ds = (
	tf.data.Dataset
	.from_tensor_slices((X_val, Y_val))
	.batch(32)
	.prefetch(AUTO)
)

model = keras.Sequential([
	layers.Dense(256, activation='relu', input_shape=[6]),
	layers.BatchNormalization(),
	layers.Dense(256, activation='relu'),
	layers.Dropout(0.3),
	layers.BatchNormalization(),
	layers.Dense(1, activation='relu')
])

model.compile(
	loss='mae',
	optimizer='adam',
	metrics=['mape']
)

model.summary()

history = model.fit(train_ds,
					epochs=50,
					validation_data=val_ds)

history_df = pd.DataFrame(history.history)
history_df.head()

history_df.loc[:, ['loss', 'val_loss']].plot()
history_df.loc[:, ['mape', 'val_mape']].plot()
plt.show()

# Convert history to DataFrame
history_df = pd.DataFrame(history.history)

