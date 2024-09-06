import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

# Loading the Data 

df = pd.read_csv('../../data/raw/Ames_Housing_Data.csv')
df.info()
df = df.drop(['PID'], axis=1)

# Data processing and transformation

df.isnull().sum()

# Create a function that reports back the percentage of missing values 

def percent_missing(df): 
    percent_nan = 100 * df.isnull().sum() / len(df)
    percent_nan = percent_nan[percent_nan > 0].sort_values()
    
    return percent_nan

percent_nan = percent_missing(df)
percent_nan.sort_values().plot(kind='bar')

# Analyze the columns with less than 1% missing values

missing_less_than_1pct = percent_nan[percent_nan < 1].index 

#Index(['Electrical', 'Garage Cars', 'BsmtFin SF 1', 'Garage Area',
#       'BsmtFin SF 2', 'Bsmt Unf SF', 'Total Bsmt SF', 'Bsmt Half Bath',
#       'Bsmt Full Bath', 'Mas Vnr Area']

# Dropping columns with less than 1% missing values, starting from smallest.

df = df.dropna(axis=0, subset=['Electrical', 'Garage Cars'])
percent_nan = percent_missing(df)

# Check again on columns with less than 1% missing values
percent_nan[percent_nan < 1].index

#Index(['Bsmt Unf SF', 'Total Bsmt SF', 'BsmtFin SF 2', 'BsmtFin SF 1',
#      'Bsmt Full Bath', 'Bsmt Half Bath', 'Mas Vnr Area'],
#      dtype='object')

# BSMT features seem to be related, let's check if they have the same number of missing values

df[df['Bsmt Half Bath'].isnull()] # index 1341 and 1497
df[df['Bsmt Full Bath'].isnull()] # index 1341 and 1497
df[df['Bsmt Unf SF'].isnull()] # index 1341 

# Following documentation of the data set, houses without basements have NaN values in these columns.
# Let's fill these NaN values with 0 or 'Nan'

# BMST NUMERIC COLUMNS -> fill values as 0
bsmt_num_cols = ['Bsmt Half Bath', 'Bsmt Full Bath', 
                 'Bsmt Unf SF', 'Total Bsmt SF',
                 'BsmtFin SF 2', 'BsmtFin SF 1']

df[bsmt_num_cols] = df[bsmt_num_cols].fillna(0)

# BMST STRING COLUMNS -> fill values as 'None'

bsmt_str_cols = ['Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 
                 'BsmtFin Type 1', 'BsmtFin Type 2']
df[bsmt_str_cols] = df[bsmt_str_cols].fillna('None')

# Lets check again for missing values 

percent_nan = percent_missing(df)
percent_nan[percent_nan < 1] # Just missing values in Mas Vnr Area

# Mas Vnr Area is the area of masonry veneer in square feet. MasVnrType is marked as NaN
# for houses without masonry veneer. Let's fill in missing values on MasVnrType with 'None'
# and missing values on MasVnrArea with 0

df['Mas Vnr Type'] = df['Mas Vnr Type'].fillna('None')
df['Mas Vnr Area'] = df['Mas Vnr Area'].fillna(0)

# Lets check again for missing values below 1% of the total data

percent_nan = percent_missing(df) # all missing percentages are above 1% now. 
percent_nan.sort_values().plot(kind='bar')

#------------------------------------------------------------

# Dealing with Missing Data above 1% threshold. - Feature Columns

# It seems like Garage type is missing the same number of values as Garage Finish, Garage Qual, Garage Cond

gar_str_cols = ['Garage Type', 'Garage Finish', 'Garage Qual', 'Garage Cond']
df[gar_str_cols] = df[gar_str_cols].fillna('None')

# Filling in missing values for Garage Yr Blt with 0

df['Garage Yr Blt'] = df['Garage Yr Blt'].fillna(0)

percent_nan = percent_missing(df)
percent_nan.sort_values().plot(kind='bar') 

# Since we're missing a lot of data for Pool QC, fence, alley and Misc Features we can drop these columns. 

df = df.drop(['Pool QC','Fence','Alley','Misc Feature'], axis=1)

# Now we have only a few columns with missing values. 

df['Fireplace Qu'].value_counts()

# It seems like this column holds string data, let's fill in missing values with 'None'

df['Fireplace Qu'] = df['Fireplace Qu'].fillna('None')

# Now we only have lot Frontage Missing. 
# LotFrontage: Linear feet of street connected to property
# We can fill in missing values with the mean of the column.

df['Lot Frontage'] = df['Lot Frontage'].fillna(df['Lot Frontage'].mean())

# Now let's check one last time for missing values. 

df.info() 
df.isnull().sum()

#------------------------------------------------------------

# Export the cleaned data to a pickle file. 

df.to_pickle('../../data/interim/01_Ames_Housing_Data_Cleaned.pkl')


