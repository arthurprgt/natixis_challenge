import pandas as pd 
import numpy as np

def complete_nan_values(df):

    df_unique_isin = df.groupby('ISIN').first()
    columns = ['Classification', 'SpreadvsBenchmarkMid', 'MidASWSpread', 'MidZSpread', 'GSpreadMid', 
               'MidModifiedDuration', 'MidConvexity', 'MidEffectiveDuration', 'MidEffectiveConvexity', 'Year_dealdate', 'Month_dealdate']
    df_by_classification = df_unique_isin[columns].copy()
    df_by_classification = df_by_classification.groupby(['Classification', 'Year_dealdate']).mean().reset_index()

    df_group_by_industry = df_by_classification.groupby('Classification').mean().reset_index()
    numeric_columns = ['SpreadvsBenchmarkMid', 'MidASWSpread', 'MidZSpread', 'GSpreadMid', 
                       'MidModifiedDuration', 'MidConvexity', 'MidEffectiveDuration', 'MidEffectiveConvexity']
    
    df_by_classification['additional_column'] = df_by_classification['Classification'].astype(str) + ' - ' + df_by_classification['Year_dealdate'].astype(str)
    df['additional_column'] = df['Classification'].astype(str) + ' - ' + df['Year_dealdate'].astype(str)

    for column in numeric_columns:
        df_by_classification[column] = df_by_classification[column].fillna(df_by_classification['Classification'].map(df_group_by_industry.set_index('Classification')[column]))

    for column in numeric_columns:
        df[column] = df[column].fillna(df['additional_column'].map(df_by_classification.set_index('additional_column')[column]))

    df.drop(columns=['additional_column'], inplace=True)
    
    return df


def preprocess_clustering(df, cols_to_exclude):

    # Drop the columns that we exclude
    df = df.drop(cols_to_exclude, axis=1, errors='ignore')

    # Identify numerical columns
    numerical_columns = df.select_dtypes(include=['number']).columns

    # Transform 'Ccy' to 'is_euro' boolean column
    df['is_euro'] = (df['Ccy'] == 'EUR').astype(int)
    # Transform 'Type' to 'is_fixed' boolean column
    df['is_fixed'] = (df['Type'] == 'Fixed').astype(int)
    # Drop the original 'Ccy' and 'Type' columns
    df = df.drop(['Ccy', 'Type'], axis=1, errors='ignore')

    # Ordinal encoding for 'Rating_Fitch'
    rating_mapping = {
        'AAA': 22,
        'AA+': 21,
        'AA': 20,
        'AA-': 19,
        'A+': 18,
        'A': 17,
        'A-': 16,
        'BBB+': 15,
        'BBB': 14,
        'BBB-': 13,
        'BB+': 12,
        'BB': 11,
        'BB-': 10,
        'B+': 9,
        'B': 8,
        'B-': 7,
        'CCC+': 6,
        'CCC': 5,
        'CCC-': 4,
        'CC': 3,
        'C': 2,
        'WD': 1,
        'D': 0,
        'NR': np.nan
    }

    rating_mapping_moodys = {
        'Aaa': 22,
        'Aa1': 21,
        'Aa2': 20,
        '(P)Aa2': 20,
        'Aa3': 19,
        '(P)Aa3': 19,
        'A1': 18,
        '(P)A1': 18,
        'A2': 17,
        '(P)A2': 17,
        'A3': 16,
        '(P)A3': 16,
        'Baa1': 15,
        '(P)Baa1': 15,
        'Baa2': 14,
        '(P)Baa2': 14,
        'Baa3': 13,
        'Ba1': 12,
        'Ba2': 11,
        'Ba3': 10,
        'B1': 9,
        'B2': 8,
        'B3': 7,
        'Caa1': 6,
        'Caa2': 5,
        'Caa3': 4,
        'Ca': 2.5,
        'C': 0
    }

    df['Rating_Fitch_encoded'] = df['Rating_Fitch'].map(rating_mapping)
    df['Rating_SP_encoded'] = df['Rating_SP'].map(rating_mapping)
    df['Rating_Moodys_encoded'] = df['Rating_Moodys'].map(rating_mapping_moodys)
    # Create a unique Rating that averages the 3 Ratings and ignores missing values
    df['Rating'] = df[['Rating_Fitch_encoded', 'Rating_SP_encoded', 'Rating_Moodys_encoded']].mean(axis=1)

    # Map values in 'Country' column
    valid_countries = ['FRANCE', 'ITALY', 'GERMANY', 'NETHERLANDS', 'SPAIN']
    df['Country'] = df['Country'].apply(lambda x: x if x in valid_countries else 'OTHER')
    # Perform one-hot encoding
    df = pd.get_dummies(df, columns=['Country'], prefix='is')

    # Map values in 'Classification' column
    valid_classes = ['Financials', 'Government', 'Industrials', 'Utilities']
    df['Classification'] = df['Classification'].apply(lambda x: x if x in valid_classes else 'OTHER')
    # Perform one-hot encoding
    df = pd.get_dummies(df, columns=['Classification'], prefix='is')

    # Add newly created boolean columns and 'Rating' to agg_dict with average
    agg_dict = {col: 'mean' for col in ['is_euro', 'is_fixed', 'Rating']}
    agg_dict.update({col: 'first' for col in ['is_FRANCE', 'is_ITALY', 'is_GERMANY', 'is_NETHERLANDS', 'is_SPAIN']})
    agg_dict.update({col: 'first' for col in ['is_Financials', 'is_Government', 'is_Industrials', 'is_Utilities']})
    agg_dict.update({num_col: ['min', 'max', 'median'] for num_col in numerical_columns})

    # Grouping by 'ISIN' and aggregating columns
    grouped_df = df.groupby('ISIN').agg(agg_dict).reset_index()

    # Flatten the multi-level column index
    grouped_df.columns = ['_'.join(col).strip() for col in grouped_df.columns.values]

    # Drop identical columns
    grouped_df = grouped_df.T.drop_duplicates().T

    return grouped_df

