import streamlit as st
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def extract_data_from_games(games, date):
    pattern = r'([A-Za-z]+|S)\s-\s([A-Za-z]+|L)\s(\d{1,2}:\d{1,2})'
    matches = re.findall(pattern, games)
    
    processed_data = [[m[0], m[1], int(m[2].split(':')[0]), int(m[2].split(':')[1])] for m in matches]
    
    df = pd.DataFrame(processed_data, columns=["First Name", "Second Name", "First Score", "Second Score"])
        
    for i in ["First Name", "Second Name"]:
        df[i] = np.where(df[i].str.startswith("S"), "Simon", np.where(df[i].str.startswith("F"), "Friedemann", "Lucas"))

    df["date"] = date
    return df

def calculate_player_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate individual player statistics including wins and total score.

    Args:
    - df (pd.DataFrame): Dataframe containing match results.

    Returns:
    - pd.DataFrame: A dataframe containing player statistics.
    """
     # Wins for the first player
    df['First Win'] = df['First Score'] > df['Second Score']

    # Wins for the second player
    df['Second Win'] = df['Second Score'] > df['First Score']

    # Calculate individual stats
    players_stats = pd.concat([
        df.groupby('First Name').agg({'First Win':'sum', 'First Score':'sum'}).rename(columns={'First Win':'Wins', 'First Score':'Total Score'}),
        df.groupby('Second Name').agg({'Second Win':'sum', 'Second Score':'sum'}).rename(columns={'Second Win':'Wins', 'Second Score':'Total Score'})
    ]).groupby(level=0).sum().sort_values("Wins", ascending=False)
    
    return players_stats
    

def calculate_combination_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate statistics based on player combinations.

    Args:
    - df (pd.DataFrame): Dataframe containing match results.

    Returns:
    - pd.DataFrame: A dataframe containing combination statistics.
    """
    df['Player Combo'] = df.apply(lambda x: tuple(sorted([x['First Name'], x['Second Name']])), axis=1)
    df['Score Combo'] = df.apply(lambda x: (x['First Score'], x['Second Score']) if x['First Name'] < x['Second Name'] else (x['Second Score'], x['First Score']), axis=1)
    df['Winner'] = df.apply(lambda x: x['First Name'] if x['First Score'] > x['Second Score'] else x['Second Name'], axis=1)

    # Calculate stats by combination
    combination_stats = df.groupby('Player Combo').apply(lambda x: pd.Series({
        'Total Score A': x['Score Combo'].str[0].sum(),
        'Total Score B': x['Score Combo'].str[1].sum(),
        'Wins A': (x['Winner'] == x['Player Combo'].str[0]).sum(),
        'Wins B': (x['Winner'] == x['Player Combo'].str[1]).sum(),
        'Balance': (x['Winner'] == x['Player Combo'].str[0]).sum() - (x['Winner'] == x['Player Combo'].str[1]).sum()
    })).sort_values("Balance", ascending=False)

    df_combination_stats = pd.DataFrame(combination_stats)
    return df_combination_stats

def calculate_streaks(results):
    def max_streak(seq, value):
        from itertools import groupby
        return max((sum(1 for _ in group) for key, group in groupby(seq) if key == value), default=0)
    
    streak_data = {
        player: {
            'longest_win_streak': max_streak(res, 1),
            'longest_loss_streak': -max_streak(res, -1)
        } for player, res in results.items()
    }
    
    return pd.DataFrame(streak_data)

def plot_individual_charts(results):
    for player, res in results.items():
        plt.figure(figsize=(10, 4))
        plt.plot(np.cumsum(res), label=f'{player} Trend')
        plt.title(f"Win/Loss Trend for {player} Over Time")
        plt.xlabel('Games')
        plt.ylabel('Cumulative Score')
        plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
        plt.legend()
        st.pyplot(plt)  # Display figure in Streamlit app
        plt.clf()  # Clear current figure

    for player, res in results.items():
        plt.figure(figsize=(10, 4))
        plt.plot(res, label=player, marker='o', linestyle='-')
        plt.title(f"Wins and Losses for {player} Over Time")
        plt.xlabel('Games')
        plt.ylabel('Result')
        plt.yticks([-1, 1], ['Loss', 'Win'])
        plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
        plt.legend()
        st.pyplot(plt)  # Display figure in Streamlit app
        plt.clf()  # Clear current figure

def derive_results(df):
    results = {
        'Simon': [],
        'Friedemann': [],
        'Lucas': []
    }

    for _, row in df.iterrows():
        if row['First Score'] > row['Second Score']:
            results[row['First Name']].append(1)  # Win for First Name
            results[row['Second Name']].append(-1)  # Loss for Second Name
        elif row['First Score'] < row['Second Score']:
            results[row['First Name']].append(-1)  # Loss for First Name
            results[row['Second Name']].append(1)  # Win for Second Name
        else:
            results[row['First Name']].append(0)  # Tie for First Name
            results[row['Second Name']].append(0)  # Tie for Second Name

    return results


df_sheet = pd.read_csv(st.secrets["public_gsheets_url"])

df_sheet['parsed_sheet_df'] = df_sheet.apply(lambda x: extract_data_from_games(x["games"], x["date"]), axis=1)
df = pd.DataFrame()
for _, row in df_sheet.iterrows():
    df = pd.concat([df, row["parsed_sheet_df"]])

# Derive player and combination stats
players_stats = calculate_player_stats(df)
combination_stats = calculate_combination_stats(df)

# Derive results
results = derive_results(df)

# Calculate win and loss streaks
streaks = calculate_streaks(results)

streaks
players_stats
combination_stats

plot_individual_charts(results)
