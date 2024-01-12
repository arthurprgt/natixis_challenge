import pandas as pd
import numpy as np


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
    df["maturity"].replace({pd.NaT: np.nan}, inplace=True)

    # Convert 'Deal_Date', 'maturity', 'AssumedMaturity', 'YTWDate' to datetime
    df["Deal_Date"] = pd.to_datetime(df["Deal_Date"])
    df["maturity"] = pd.to_datetime(
        df["maturity"], errors="coerce", format="%Y-%m-%d %H:%M:%S.%f"
    )
    df["AssumedMaturity"] = pd.to_datetime(df["AssumedMaturity"], errors="coerce")
    df["YTWDate"] = pd.to_datetime(df["YTWDate"], errors="coerce")

    # Add year, month, day for clustering
    df["Year_dealdate"] = df["Deal_Date"].dt.year
    df["Month_dealdate"] = df["Deal_Date"].dt.month
    df["Day_dealdate"] = df["Deal_Date"].dt.day
    df["Year_maturity"] = df["maturity"].dt.year
    df["Month_maturity"] = df["maturity"].dt.month
    df["Day_maturity"] = df["maturity"].dt.day

    # Delete maturities smaller than 2021 (as deal dates starts in 2021)
    df = df[df["maturity"].dt.year >= 2021]

    # Compute number of days between maturity and deal date
    df["Days_to_Maturity"] = (df["maturity"] - df["Deal_Date"]).dt.days

    # Replace null values in 'AssumedMaturity' with values from 'Maturity'
    df["AssumedMaturity"] = df["AssumedMaturity"].fillna(df["Maturity"])

    # Convert 'B_Side' column to boolean (1 for 'NATIXIS BUY', 0 for 'NATIXIS SELL')
    df = df[df["B_Side"].isin(["NATIXIS SELL", "NATIXIS BUY"])]
    df["B_Side"] = df["B_Side"].replace({"NATIXIS BUY": 1, "NATIXIS SELL": 0})

    # Convert null values of 'Tier'
    df["Tier"].fillna("UNKNOWN", inplace=True)

    # Lower string names
    df["Sales_Name"] = df["Sales_Name"].str.lower()
    df["company_short_name"] = df["company_short_name"].str.lower()

    # Drop unused columns
    columns_to_drop = ["Cusip", "Maturity"]
    df.drop(columns=columns_to_drop, inplace=True)

    return df
