import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate
import matplotlib.pyplot as plt

def plot_double_general_info(df, ISIN1, ISIN2):
    """
    Generate a formatted table with general information for two given ISINs in the DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing bond data.
    - ISIN1 (str): The first ISIN code for analysis.
    - ISIN2 (str): The second ISIN code for analysis.

    Returns:
    - str: Formatted table with general information for the specified ISINs.
    """ 
    # Filter DataFrame for ISIN1 and ISIN2
    df_ISIN1 = df[df['ISIN'] == ISIN1].copy()
    df_ISIN1.sort_values(by='Deal_Date', inplace=True)
    df_ISIN2 = df[df['ISIN'] == ISIN2].copy()
    df_ISIN2.sort_values(by='Deal_Date', inplace=True)

    # Check if the DataFrames are empty after filtering
    if df_ISIN1.empty or df_ISIN2.empty:
        return "No data found for the given ISINs."
    
    # Format the maturity date for ISIN1
    maturity_date1 = df_ISIN1.maturity.iloc[0].strftime("%Y-%m-%d %H:%M:%S")
    maturity_date1 = maturity_date1[:-9]  
    price1 = df_ISIN1['B_Price'].iloc[-1]

    # Format the maturity date for ISIN2
    maturity_date2 = df_ISIN2.maturity.iloc[0].strftime("%Y-%m-%d %H:%M:%S")
    maturity_date2 = maturity_date2[:-9]  
    price2 = df_ISIN2['B_Price'].iloc[-1]

    # Create a list of lists with data for both ISINs
    data = [
        [str(df_ISIN1.ISIN.iloc[0]) + ' (Initial)',
         df_ISIN1.cdcissuer.iloc[0],
         df_ISIN1.Country.iloc[0],
         df_ISIN1.BloomIndustrySector.iloc[0],
         df_ISIN1.BloomIndustrySubGroup.iloc[0],
         maturity_date1,
         price1,
         np.round(df_ISIN1.Coupon.iloc[0], 2),
         df_ISIN1.Frequency.iloc[0],
        ],
        
        [df_ISIN2.ISIN.iloc[0]+ ' (Recommended)',
         df_ISIN2.cdcissuer.iloc[0],
         df_ISIN2.Country.iloc[0],
         df_ISIN2.BloomIndustrySector.iloc[0],
         df_ISIN2.BloomIndustrySubGroup.iloc[0],
         maturity_date2,
         price2,
         np.round(df_ISIN2.Coupon.iloc[0], 2),
         df_ISIN2.Frequency.iloc[0],
        ]
    ]

    # Create a fancy grid table using tabulate
    table = tabulate(data, headers=["ISIN code", "Issuer", "Country", "Sector", "Industry Subgroup", 
                                    "Maturity", "Price (EUR)", "Coupon", "Frequency"],
                     tablefmt='fancy_grid', numalign="center", stralign="center", colalign=("center",),
                     showindex=False)

    # Print the table
    print(table)


def plot_rating_gauges(df, ISIN1, ISIN2, rating_name='Fitch'):
    """
    Generate side-by-side rating gauge plots for two specified ISINs based on the selected credit rating agency.

    Parameters:
    - df (pd.DataFrame): DataFrame containing bond data.
    - ISIN1 (str): The ISIN code of the initial bond.
    - ISIN2 (str): The ISIN code of the recommended bond.
    - rating_name (str, optional): The credit rating agency name (default is 'Fitch').
    
    Returns:
    - None: Displays the rating gauge plots using Matplotlib.

    Note:
    - The rating_name parameter can be set to 'Fitch', 'Moodys', or 'SP' to visualize ratings from Fitch, Moody's, or Standard & Poor's.
    """
    # Dataframes for both ISINs
    df_ISIN1 = df[df['ISIN'] == ISIN1]
    df_ISIN2 = df[df['ISIN'] == ISIN2]

    if rating_name == 'Fitch':
        sorted_ratings = ['WD', 'B+', 'BB-', 'BB', 'BB+', 'BBB-', 'BBB', 'BBB+', 'A-', 'A', 'A+', 'AA', 'AA+', 'AAA']
        grade1 = df_ISIN1.Rating_Fitch.iloc[0]
        grade2 = df_ISIN2.Rating_Fitch.iloc[0]

    if rating_name == 'Moodys':
        sorted_ratings = ['NR', 'WR', 'Caa1', 'B3', 'B2', 'B1', 'Ba3', 'Ba2', 'Ba1', 'Baa3', 'Baa2', 'Baa1', 'A3', 'A2', 'A1', 'Aa3', 'Aa2', 'Aa1', 'Aaa']
        grade1 = df_ISIN1.Rating_Moodys.iloc[0]
        grade2 = df_ISIN2.Rating_Moodys.iloc[0]

    if rating_name == 'SP':
        sorted_ratings = ['NR', 'CCC+', 'CC', 'B+', 'BB', 'BB+', 'BB-', 'BBB-', 'BBB', 'BBB+', 'A-', 'A', 'A+', 'AA-', 'AA', 'AA+', 'AAA']
        grade1 = df_ISIN1.Rating_SP.iloc[0]
        grade2 = df_ISIN2.Rating_SP.iloc[0]

    # Create a figure and axes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 2.1))

    # Plot for ISIN1
    cmap = plt.get_cmap('PRGn')  # Modified colormap for red to green gradient
    colors1 = [cmap(i / len(sorted_ratings)) for i in range(len(sorted_ratings))]
    for i, rating in enumerate(sorted_ratings):
        ax1.axvspan(i, i + 1, facecolor=colors1[i], alpha=0.8)

    if grade1 and grade1 in sorted_ratings:
        grade_index1 = sorted_ratings.index(grade1)
        ax1.axvline(x=grade_index1 + 0.5, color='red', linestyle='-', linewidth=3)

    ax1.set_xticks(np.arange(len(sorted_ratings)) + 0.5)
    ax1.set_xticklabels(sorted_ratings, fontsize=9)
    ax1.set_yticks([])
    ax1.set_aspect('auto')
    ax1.set_title(f'{rating_name} rating of initial ISIN ({ISIN1})')

    # Plot for ISIN2
    colors2 = [cmap(i / len(sorted_ratings)) for i in range(len(sorted_ratings))]
    for i, rating in enumerate(sorted_ratings):
        ax2.axvspan(i, i + 1, facecolor=colors2[i], alpha=0.8)

    if grade2 and grade2 in sorted_ratings:
        grade_index2 = sorted_ratings.index(grade2)
        ax2.axvline(x=grade_index2 + 0.5, color='red', linestyle='-', linewidth=3)

    ax2.set_xticks(np.arange(len(sorted_ratings)) + 0.5)
    ax2.set_xticklabels(sorted_ratings, fontsize=9)
    ax2.set_yticks([])
    ax2.set_aspect('auto')
    ax2.set_title(f'{rating_name} rating of recommended ISIN ({ISIN2})')
    plt.tight_layout()
    plt.show()

def plot_double_clients_repartition(df, ISIN_ex1, ISIN_ex2):
    """
    Generate side-by-side pie charts depicting the distribution of clients for two specified ISINs.

    Parameters:
    - df (pd.DataFrame): DataFrame containing bond data.
    - ISIN_ex1 (str): The ISIN code of the initial bond.
    - ISIN_ex2 (str): The ISIN code of the recommended bond.

    Returns:
    - None: Displays side-by-side pie charts using Matplotlib and Seaborn.

    Note:
    - The function calculates the percentage distribution of clients for each ISIN and aggregates minor clients
      into an 'Others' category for better visualization.
    """    
    # Generate the dataframe for ISIN_ex1
    df_ISIN1 = df[df['ISIN'] == ISIN_ex1].copy()
    clien_unique_count1 = df_ISIN1['company_short_name'].value_counts()
    client_count_df1 = pd.DataFrame(clien_unique_count1.reset_index())
    total_count1 = client_count_df1['count'].sum()
    client_count_df1['count_perc'] = np.round(client_count_df1['count'] * 100 / total_count1, 2)
    client_count_df1.drop(columns=['count'], inplace=True)
    df_client_other1 = client_count_df1[client_count_df1['count_perc'] < 4].copy()
    client_count_df1 = client_count_df1[client_count_df1['count_perc'] >= 4].copy()
    count_perc_other1 = np.round(df_client_other1.count_perc.sum(), 2)

    # Add the 'Other' line to the DataFrame for ISIN_ex1
    line_to_add1 = pd.DataFrame({'company_short_name': ['Others'], 'count_perc': [count_perc_other1]})
    client_count_df1 = pd.concat([client_count_df1, line_to_add1], ignore_index=True, axis=0)

    # Generate the dataframe for ISIN_ex2
    df_ISIN2 = df[df['ISIN'] == ISIN_ex2].copy()
    clien_unique_count2 = df_ISIN2['company_short_name'].value_counts()
    client_count_df2 = pd.DataFrame(clien_unique_count2.reset_index())
    total_count2 = client_count_df2['count'].sum()
    client_count_df2['count_perc'] = np.round(client_count_df2['count'] * 100 / total_count2, 2)
    client_count_df2.drop(columns=['count'], inplace=True)
    df_client_other2 = client_count_df2[client_count_df2['count_perc'] < 6].copy()
    client_count_df2 = client_count_df2[client_count_df2['count_perc'] >= 6].copy()
    count_perc_other2 = np.round(df_client_other2.count_perc.sum(), 2)

    # Add the 'Other' line to the DataFrame for ISIN_ex2
    line_to_add2 = pd.DataFrame({'company_short_name': ['Others'], 'count_perc': [count_perc_other2]})
    client_count_df2 = pd.concat([client_count_df2, line_to_add2], ignore_index=True, axis=0)

    # Plotting pie charts side by side
    plt.figure(figsize=(20, 6))

    # Plot for ISIN_ex1
    plt.subplot(1, 2, 1)
    colors1 = sns.color_palette("rainbow", len(client_count_df1))
    plt.pie(client_count_df1['count_perc'], autopct='%1.1f%%', colors=colors1)
    plt.title(f'Distribution of Clients for Initial ISIN ({ISIN_ex1})', fontsize=18)
    plt.legend(client_count_df1['company_short_name'], bbox_to_anchor=(0.95, 0.95), 
               loc='upper left', title='Company Names')

    # Plot for ISIN_ex2
    plt.subplot(1, 2, 2)
    colors2 = sns.color_palette("rainbow", len(client_count_df2))
    plt.pie(client_count_df2['count_perc'], autopct='%1.1f%%', colors=colors2)
    plt.title(f'Distribution of Clients for Recommended ISIN ({ISIN_ex2})', fontsize=18)
    plt.legend(client_count_df2['company_short_name'], bbox_to_anchor=(0.95, 0.95), 
               loc='upper left', title='Company Names')

    plt.show()

def plot_monthly_traded_volume(df, ISIN1, ISIN2):
    """
    Plots the monthly traded volume for two given ISINs side by side.

    Parameters:
    - df (DataFrame): The DataFrame containing bond data.
    - ISIN1 (str): The International Securities Identification Number of the first bond.
    - ISIN2 (str): The International Securities Identification Number of the second bond.

    Returns:
    None
    """
    # Process the dataframe for ISIN1
    df_ISIN1 = df[df['ISIN'] == ISIN1].copy()
    df_ISIN1['Deal_Date'] = pd.to_datetime(df_ISIN1['Deal_Date'])
    df_ISIN1['Date'] = df_ISIN1['Deal_Date'].dt.to_period('M')
    df_ISIN1 = df_ISIN1.groupby('Date')['Total_Traded_Volume'].sum().reset_index()

    # Process the dataframe for ISIN2
    df_ISIN2 = df[df['ISIN'] == ISIN2].copy()
    df_ISIN2['Deal_Date'] = pd.to_datetime(df_ISIN2['Deal_Date'])
    df_ISIN2['Date'] = df_ISIN2['Deal_Date'].dt.to_period('M')
    df_ISIN2 = df_ISIN2.groupby('Date')['Total_Traded_Volume'].sum().reset_index()

    # Plot side by side
    plt.figure(figsize=(16, 5))

    # Plot for ISIN1
    plt.subplot(1, 2, 1)
    ax1 = sns.barplot(x='Date', y='Total_Traded_Volume', data=df_ISIN1, color='purple')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.title(f'Monthly Traded Volume for Initial ISIN: {ISIN1}\n', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Total Volume Traded')

    # Plot for ISIN2
    plt.subplot(1, 2, 2)
    ax2 = sns.barplot(x='Date', y='Total_Traded_Volume', data=df_ISIN2, color='purple')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    plt.title(f'Monthly Traded Volume for Recommended ISIN: {ISIN2}\n', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Total Volume Traded')

    plt.tight_layout()
    plt.show()

def plot_monthly_transactions_volume(df, ISIN1, ISIN2):
    """
    Plots the monthly traded number of transactions for two given ISINs side by side.

    Parameters:
    - df (DataFrame): The DataFrame containing bond data.
    - ISIN1 (str): The International Securities Identification Number of the first bond.
    - ISIN2 (str): The International Securities Identification Number of the second bond.

    Returns:
    None
    """
    # Process the dataframe for ISIN1
    df_ISIN1 = df[df['ISIN'] == ISIN1].copy()
    df_ISIN1['Deal_Date'] = pd.to_datetime(df_ISIN1['Deal_Date'])
    df_ISIN1['Date'] = df_ISIN1['Deal_Date'].dt.to_period('M')
    df_ISIN1['transaction'] = 1
    df_ISIN1 = df_ISIN1.groupby('Date')['transaction'].sum().reset_index()

    # Process the dataframe for ISIN2
    df_ISIN2 = df[df['ISIN'] == ISIN2].copy()
    df_ISIN2['Deal_Date'] = pd.to_datetime(df_ISIN2['Deal_Date'])
    df_ISIN2['Date'] = df_ISIN2['Deal_Date'].dt.to_period('M')
    df_ISIN2['transaction'] = 1
    df_ISIN2 = df_ISIN2.groupby('Date')['transaction'].sum().reset_index()

    # Plot side by side
    plt.figure(figsize=(16, 5))

    # Plot for ISIN1
    plt.subplot(1, 2, 1)
    ax1 = sns.barplot(x='Date', y='transaction', data=df_ISIN1, color='purple')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.title(f'Monthly RFQs for Initial ISIN: {ISIN1}\n', fontsize=15)
    plt.xlabel('Date')
    plt.ylabel('Total RFQs')

    # Plot for ISIN2
    plt.subplot(1, 2, 2)
    ax2 = sns.barplot(x='Date', y='transaction', data=df_ISIN2, color='purple')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    plt.title(f'Monthly RFQs Volume for Recommended ISIN: {ISIN2}\n', fontsize=15)
    plt.xlabel('Date')
    plt.ylabel('Total RFQs')

    plt.tight_layout()
    plt.show()

def plot_double_price_evolution(df, ISIN1, ISIN2):
    """
    Plots the price evolution of a bond over its deal dates.

    Parameters:
    - df (DataFrame): The DataFrame containing bond data.
    - ISIN (str): The International Securities Identification Number of the bond.

    Returns:
    None
    """
    # Process the dataframe
    df_ISIN1 = df[df['ISIN'] == ISIN1].copy()
    df_ISIN1.sort_values(by='Deal_Date', inplace=True)

    df_ISIN2 = df[df['ISIN'] == ISIN2].copy()
    df_ISIN2.sort_values(by='Deal_Date', inplace=True)

    # Plot the figure with violet color scheme
    plt.figure(figsize=(12, 5))
    df_ISIN1_currency = df_ISIN1.iloc[0].Ccy
    df_ISIN2_currency = df_ISIN2.iloc[0].Ccy
    plt.plot(df_ISIN1['Deal_Date'], df_ISIN1['B_Price'], color='purple', linestyle='-', linewidth=2, marker='o', markersize=4, label=f'Initial ISIN {ISIN1} (in {df_ISIN1_currency})')
    plt.plot(df_ISIN2['Deal_Date'], df_ISIN2['B_Price'], color='purple', alpha=0.3, linestyle='-', marker='o',  markersize=4, label=f'Recommended ISIN {ISIN2} (in {df_ISIN2_currency})')
    plt.xlabel('Deal Date', fontsize=15)
    plt.ylabel(f'Bond Price', fontsize=15)
    plt.title(f'Bond Price evolution as a function of the Deal Date\n', fontsize=18)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7) 
    plt.show()
