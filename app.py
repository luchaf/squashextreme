import streamlit as st
import re
import pandas as pd
import datetime
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

def plot_individual_charts(results):
    # Define colors for the players.
    player_colors = {
        'Simon': 'red',
        'Friedemann': 'blue',
        'Lucas': 'green'
    }

    # Check the background color to determine if it's a dark theme or light theme
    bg_color = st.get_option("theme.backgroundColor")
    if bg_color is None:
        bg_color = "#FFFFFF"  # Default to light theme's white background
    
    is_dark_theme = int(bg_color[1:3], 16) < 128  # Convert hex to integer and check if it's below mid-range


    # Set title color based on theme
    title_color = 'white' if is_dark_theme else 'black'
    title_color = 'white'

    for player, res in results.items():
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Plotting the cumulative sum of results
        ax.plot(np.cumsum(res), color=player_colors[player], marker='o', linestyle='-')
        
        # Titles, labels, and legends
        ax.set_title(f"Win/Loss Trend for {player} Over Time", fontsize=16, color=title_color)
        ax.set_xlabel('Games', fontsize=14, color=title_color)
        ax.set_ylabel('Cumulative Score', fontsize=14, color=title_color)
        
        # Making the background transparent and removing unwanted lines
        fig.patch.set_alpha(0.0)
        ax.set_facecolor((0, 0, 0, 0))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='x', colors=title_color)
        ax.tick_params(axis='y', colors=title_color)
        ax.grid(False)  # Turn off grid
        
        st.pyplot(fig, transparent=True)

    for player, res in results.items():
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Plotting wins and losses over time
        ax.plot(res, marker='o', linestyle='-', color=player_colors[player])
        
        # Titles, labels, and legends
        ax.set_title(f"Wins and Losses for {player} Over Time", fontsize=16, color=title_color)
        ax.set_xlabel('Games', fontsize=14, color=title_color)
        ax.set_ylabel('Result', fontsize=14, color=title_color)
        ax.set_yticks([-1, 1])
        ax.set_yticklabels(['Loss', 'Win'], color=title_color)
        
        # Making the background transparent and removing unwanted lines
        fig.patch.set_alpha(0.0)
        ax.set_facecolor((0, 0, 0, 0))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='x', colors=title_color)
        ax.tick_params(axis='y', colors=title_color)
        ax.grid(False)  # Turn off grid
        
        st.pyplot(fig, transparent=True)

df_sheet = pd.read_csv(st.secrets["public_gsheets_url"])
df_sheet["date"] = df_sheet["date"].astype(str)
list_of_available_dates = df_sheet["date"].tolist()
selected_items = st.multiselect('Choose one or several matchday(s):', list_of_available_dates, placeholder="ask and thou ball receive")
#start_date = st.date_input("Start date:")
#st.write('Day to document:', start_date)
#end_date = st.date_input("End date:", datetime.datetime.now())
#st.write('Day to document:', end_date)

df_sheet['parsed_sheet_df'] = df_sheet.apply(lambda x: extract_data_from_games(x["games"], x["date"]), axis=1)
df_tmp = pd.DataFrame()
for _, row in df_sheet.iterrows():
    df_tmp = pd.concat([df_tmp, row["parsed_sheet_df"]])
#df['date'] = pd.to_datetime(df['date'], format='%Y%m%d').dt.date
#df = df[(df["date"]>=start_date) & (df["date"]<=end_date)].copy()
df = df_tmp[df_tmp["date"].isin(selected_items)].copy()

if df.empty!=True:
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
