import pandas as pd 
import numpy as np
from utils import complete_nan_values 
from utils import preprocess_clustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

# Load preprocessed data
df_preprocessed = pd.read_csv("data/preprocessed_data.csv")

# Fill missing values in key financial columns
df_filled = complete_nan_values(df_preprocessed)

# Featurize and group by ISIN number
cols_to_exclude = ['Deal_Date', 'cusip', 'B_Side', 'Instrument', 'Sales_Name', 'Sales_Initial', 'company_short_name',
                   'Total_Requested_Volume', 'Total_Traded_Volume_Natixis', 'Total_Traded_Volume_Away', 'Total_Traded_Volume',
                   'cdcissuer', 'Tier', 'Year_dealdate', 'Month_dealdate','Day_dealdate', 'Days_to_Maturity',
                   'cdcissuerShortName', 'lb_Platform_2', 'Day_maturity']
df_clustering = preprocess_clustering(df_filled, cols_to_exclude)

# Fill missing values from the Rating feature
df_clustering_filled = df_clustering.copy()
df_clustering_filled['Rating_mean'] = df_clustering_filled['Rating_mean'].fillna(df_clustering['Rating_mean'].median())

# Normalize the data to prepare for distance calculations
scaler = StandardScaler()
df_normalized = df_clustering_filled.drop(columns=['ISIN_'])
df_normalized = scaler.fit_transform(df_normalized)

# Calculate distances and output 5 recommended bonds
def get_nearest_rows_with_proximity_scores(df_normalized, df_clustering_filled, isin_string):
    # Find the index of the given ISIN string in df_original
    index = df_clustering_filled[df_clustering_filled['ISIN_'] == isin_string].index[0]
    
    # Calculate Euclidean distances between the selected row and all other rows
    distances = euclidean_distances(df_normalized, [df_normalized[index]])
    
    # Get the indices of the 5 nearest rows (excluding the row itself)
    nearest_indices = np.argsort(distances.flatten())[1:6]
    
    # Retrieve the corresponding rows and distances from the original DataFrame
    nearest_rows = df_clustering_filled.iloc[nearest_indices]
    nearest_distances = distances.flatten()[nearest_indices]
    
    # Calculate proximity scores
    max_distance = np.max(distances)
    proximity_scores = 1 - nearest_distances / max_distance
    
    return nearest_rows['ISIN_'], proximity_scores
