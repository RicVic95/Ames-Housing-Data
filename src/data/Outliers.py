import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Loading the Data 

df = pd.read_pickle('../../data/processed/01_Ames_Housing_Data_Cleaned.pkl') 

# Correlation between features and target variable

df.corr(numeric_only=True)['SalePrice'].sort_values(ascending=False)

# Visualizing the data to identify outliers. 

sns.scatterplot(x='Overall Qual', y='SalePrice', data=df) # Potential Outliers on high overall qual but low price. 
sns.scatterplot(x='Gr Liv Area', y='SalePrice', data=df) # Potential Outliers on high living area but low price.

# Removing outliers with high correlation to sale price.

drop_index = df[(df['Gr Liv Area'] > 4000) & (df['SalePrice'] < 200000)].index

df = df.drop(drop_index, axis=0)

# Check scatterplot again to see if outliers were removed.

sns.scatterplot(x='Gr Liv Area', y='SalePrice', data=df)

# Save cleaned data

df.to_pickle('../../data/interim/02_Ames_Housing_Data_Outliers_Removed.pkl')





