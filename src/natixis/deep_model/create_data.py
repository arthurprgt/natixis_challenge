import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler

def preprocess_dataframe(df):
    """
    Preprocesses the input DataFrame with the following steps:
    1. Converts 'Deal_Date', 'maturity', 'AssumedMaturity', 'YTWDate' columns to datetime.
    2. Converts 'B_Side' column to boolean (1 for 'NATIXIS BUY', 0 for 'NATIXIS SELL').
    3. Converts 'B_Price' and 'Total_Requested_Volume' columns to integers.
    4. Fills null values in 'Tier', 'AssumedMaturity', and 'YTWDate' columns with 'UNKNOWN'.
    5. Converts 'Frequency' feature values into integers (removing 'M' from the end).
    6. Drops the unsused 'Cusip' column.

    Parameters:
    - df (DataFrame): Input DataFrame.

    Returns:
    - DataFrame: Processed DataFrame.
    """

    df = df.copy()

    # Drop null values only for columns below the threshold
    columns_to_delete_null_vales = [
        "MidYTM",
        "Coupon",
        "Ccy",
        "cusip",
        "maturity",
        "cdcissuerShortName",
        "Frequency",
        "MidPrice",
        "cdcissuer",
        "company_short_name",
        "BloomIndustrySubGroup",
        "B_Price",
        "Total_Traded_Volume_Natixis",
        "B_Side",
        "Total_Traded_Volume_Away",
        "Total_Requested_Volume",
        "Total_Traded_Volume",
        "Type",
        "Maturity",
        "ISIN",
        "Deal_Date",
    ]
    df = df.dropna(subset=columns_to_delete_null_vales)

    # Convert 'B_Price', 'Total_Requested_Volume', 'Frequency' to integers
    df["Frequency"] = df["Frequency"].str.replace("M", "")
    numerical_columns = ["B_Price", "Total_Requested_Volume", "Frequency"]
    df.dropna(subset=numerical_columns, inplace=True)
    for column in numerical_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype(int)

    # Fix the error in the B_Price column
    df = df[df["B_Price"] >= 20]

    # Replace NaT with null values in the 'Maturity' column
    #df["maturity"].replace({pd.NaT: np.nan}, inplace=True)

    # Convert 'Deal_Date', 'maturity', 'AssumedMaturity', 'YTWDate' to datetime
    df["Deal_Date"] = pd.to_datetime(df["Deal_Date"])
    df["Maturity"] = pd.to_datetime(df["Maturity"])
    df["AssumedMaturity"] = pd.to_datetime(df["AssumedMaturity"], errors="coerce")
    df["YTWDate"] = pd.to_datetime(df["YTWDate"], errors="coerce")

    # Add year, month, day for clustering
    df["Year_dealdate"] = df["Deal_Date"].dt.year
    df["Month_dealdate"] = df["Deal_Date"].dt.month
    df["Day_dealdate"] = df["Deal_Date"].dt.day
    df["Year_maturity"] = df["Maturity"].dt.year
    df["Month_maturity"] = df["Maturity"].dt.month
    df["Day_maturity"] = df["Maturity"].dt.day

    # Delete maturities smaller than 2021 (as deal dates starts in 2021)
    df = df[df["Maturity"].dt.year >= 2021]

    # Compute number of days between maturity and deal date
    df["Days_to_Maturity"] = (df["Maturity"] - df["Deal_Date"]).dt.days

    # Replace null values in 'AssumedMaturity' with values from 'Maturity'
    df["AssumedMaturity"] = df["AssumedMaturity"].fillna(df["Maturity"])

    # Convert 'B_Side' column to boolean (1 for 'NATIXIS BUY', 0 for 'NATIXIS SELL')
    df = df[df["B_Side"].isin(["NATIXIS SELL", "NATIXIS BUY"])]
    df["B_Side"] = df["B_Side"].replace({"NATIXIS BUY": 1, "NATIXIS SELL": 2})

    # Convert null values of 'Tier'
    df["Tier"].fillna("UNKNOWN", inplace=True)

    # Lower string names
    df["Sales_Name"] = df["Sales_Name"].str.lower()
    df["company_short_name"] = df["company_short_name"].str.lower()

    # Drop unused columns
    columns_to_drop = ["Cusip", "Maturity"]
    df.drop(columns=columns_to_drop, inplace=True)

    return df

def encode_dataframe(df):
    # Delete unnecessary columns
    index_columns = ['Deal_Date', 'ISIN', 'company_short_name']
    numerical_columns = ['B_Price', 'Coupon', 'MidPrice', 'MidYTM', 'MidASWSpread',
                        'MidZSpread', 'MidModifiedDuration', 'MidConvexity', 'MidEffectiveDuration',
                        'MidEffectiveConvexity', 'Year_maturity', 'Days_to_Maturity']
    categorical_columns = ['B_Side', 'BloomIndustrySector', 'BloomIndustryGroup', 'BloomIndustrySubGroup',
                        'cdcissuerShortName', 'Country', 'lb_Platform_2', 'Rating_Fitch',
                        'Rating_Moodys', 'Rating_SP', 'Ccy', 'Classification', 'Tier',
                        'Frequency', 'Type']
    features = index_columns + numerical_columns + categorical_columns
    df_subset = df[features]

    # Sort values
    df_subset = df_subset.sort_values(by=['ISIN', 'Deal_Date'])

    # Convert 'Deal_Date' to datetime format
    df_subset['Deal_Date'] = pd.to_datetime(df_subset['Deal_Date'])

    # Add positive signal
    df_subset['Signal'] = df_subset['B_Side']

    # Delete unknown values
    df_subset.dropna(inplace=True)

    # Reset index
    df_subset = df_subset.reset_index(drop=True)
    
    # Add negative signals
    row_0 = df_subset.iloc[0]
    previous_investors = set()

    for i, row in df_subset.iterrows():
        if row['ISIN'] == row_0['ISIN']:
            previous_investors.add(row['company_short_name'])

            if row['Deal_Date'] > row_0['Deal_Date'] + pd.Timedelta(days=5):
                not_interested = [x for x in previous_investors if x != row['company_short_name']]
                random.shuffle(not_interested)
                
                ### SEE HOW MANY NEGATIVE SIGNALS WE ADD - here 3
                for inv in not_interested[:3]:
                    df_subset.loc[len(df_subset)] = df_subset.loc[i].copy()
                    df_subset.loc[len(df_subset) - 1, 'company_short_name'] = inv
                    df_subset.loc[len(df_subset) - 1, 'Signal'] = 0

                row_0 = row

        else:
            print(str(list(df_subset.ISIN.unique()).index(row['ISIN'])) + "/" + str(len(df_subset.ISIN.unique())))
            previous_investors = set([row['company_short_name']])
            row_0 = row
    
    # Reformat the dataframe
    df_final = df_subset[["Deal_Date", "ISIN", "company_short_name", "Signal"] + numerical_columns + categorical_columns]

    # Standardize numerical columns
    scaler = StandardScaler()
    df_final[numerical_columns] = scaler.fit_transform(df_final[numerical_columns])

    # Encode categorical columns
    df_final = pd.get_dummies(df_final, columns=categorical_columns, drop_first=True, dtype=int)

    return df_final

# Loading the data
df = pd.read_csv("data/data.csv")

# Creating and saving the preprocessed dataframe for EDA
df_preprocessed = preprocess_dataframe(df)
df_preprocessed.to_csv("data/preprocessed_data_final.csv", index=False)

# Creating and saving the training data for the DL model
df_encoded = encode_dataframe(df_preprocessed)
df_encoded.to_csv("data/new_dataset_final.csv", index=False)