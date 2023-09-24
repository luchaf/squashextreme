import streamlit as st
import re
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator

def extract_data_from_games(games, date):
    pattern = r'([A-Za-z]+|S)\s-\s([A-Za-z]+|L)\s(\d{1,2}:\d{1,2})'
    matches = re.findall(pattern, games)
    
    processed_data = [[m[0], m[1], int(m[2].split(':')[0]), int(m[2].split(':')[1])] for m in matches]
    
    df = pd.DataFrame(processed_data, columns=["First Name", "Second Name", "First Score", "Second Score"])
        
    for i in ["First Name", "Second Name"]:
        df[i] = np.where(df[i].str.startswith("S"), "Simon", np.where(df[i].str.startswith("F"), "Friedemann", np.where(df[i].str.startswith("L"), "Lucas", "Lucas")))

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

color_map = {
    'Lucas': 'green',
    'Simon': 'blue',
    'Friedemann': 'pink'
}

# Define colors for the players.
player_colors = {
        'Simon': 'blue',
        'Friedemann': 'pink',
        'Lucas': 'green',
        "Leeroy Jenkins": "grey",
    }

title_color = 'black'

def win_loss_trends_plot(results):
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
        #ax.grid(False)  # Turn off grid
        plt.tight_layout()
        st.pyplot(fig, transparent=True)

def wins_and_losses_over_time_plot(results):
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
        #ax.grid(False)  # Turn off grid
        plt.tight_layout()
        st.pyplot(fig, transparent=True)


def plot_wins_and_total_scores(df2):

    # Check the background color to determine if it's a dark theme or light theme
    bg_color = st.get_option("theme.backgroundColor")
    if bg_color is None:
        bg_color = "#FFFFFF"  # Default to light theme's white background
    
    is_dark_theme = int(bg_color[1:3], 16) < 128  # Convert hex to integer and check if it's below mid-range

    # Set title color based on theme
    title_color = 'white' if is_dark_theme else 'black'
    
    # Bar configurations
    bar_width = 0.35
    r1 = np.arange(len(df2['Wins']))  # positions for Wins bars
    r2 = [x + bar_width for x in r1]  # positions for Total Score bars

    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot 'Wins'
    bars1 = ax1.bar(r1, df2['Wins'], color='blue', alpha=0.6, width=bar_width, label='Wins')
    ax1.set_ylabel('Wins', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.tick_params(axis='x', colors=title_color)
    
    # Annotations for Wins
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, int(yval), ha='center', va='bottom', color='blue')
    
    # Plot 'Total Score' on the second y-axis
    ax2 = ax1.twinx()
    bars2 = ax2.bar(r2, df2['Total Score'], color='red', alpha=0.6, width=bar_width, label='Total Score')
    ax2.set_ylabel('Total Score', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Annotations for Total Score
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval + 10, int(yval), ha='center', va='bottom', color='red')
    
    # Adjust x-tick labels to center between two bars
    ax1.set_xticks([r + bar_width / 2 for r in range(len(df2['Wins']))])
    ax1.set_xticklabels(df2.index)
    
    # Set y-ticks
    ax1_ticks = np.arange(0, df2['Wins'].max() + 5, 5)
    ax2_ticks = ax1_ticks * 20
    
    ax1.set_yticks(ax1_ticks)
    ax2.set_yticks(ax2_ticks)
    
    # Grid settings
    ax1.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6)
    ax2.grid(None)
    
    ax1.set_title('Wins and Total Scores for Players', color=title_color)
    
    # Styling settings
    fig.patch.set_alpha(0.0)
    ax1.set_facecolor((0, 0, 0, 0))
    ax2.set_facecolor((0, 0, 0, 0))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig, transparent=True)
    

def graph_win_and_loss_streaks(df1):
    # Define colors for the win and loss streaks
    colors = {'longest_win_streak': 'green', 'longest_loss_streak': 'red'}
    title_color = 'black'
    
    ax = df1.plot(kind='bar', figsize=(10, 6), color=[colors[col] for col in df1.columns])
    plt.title("Streaks for Players", fontsize=16, color=title_color)
    plt.ylabel("Number of Matches", fontsize=14, color=title_color)
    plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6)
    
    # Annotate bars with their values
    for p in ax.patches:
        ax.annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color=title_color, 
                    xytext=(0, 10 if p.get_height() > 0 else -10), 
                    textcoords='offset points')

    # Styling settings
    ax.set_facecolor((0, 0, 0, 0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='x', colors=title_color, rotation=0)
    ax.tick_params(axis='y', colors=title_color)
    
    # Ensure y-axis has integer ticks only
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    st.pyplot(plt, transparent=True)

color_map = {
    'Lucas': 'green',
    'Simon': 'blue',
    'Friedemann': 'pink'
}

def get_colors(players):
    return [color_map[player] for player in players]

def annotate_bars(ax, bars):
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, int(yval), 
                ha='center', va='bottom', color='black', fontsize=10)

def style_axes(ax, title, ylabel):
    # Set title and labels
    ax.set_title(title, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=14)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set grid
    ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6)


def plot_player_combo_wins_graph(df):
    # Bar width
    bar_width = 0.35

    # Wins Graph
    fig, ax = plt.subplots(figsize=(10, 6))
    r1 = np.arange(len(df))
    r2 = [x + bar_width for x in r1]
    
    bars1 = ax.bar(r1, df['Wins A'], width=bar_width, color=get_colors([combo[0] for combo in df.index]))
    bars2 = ax.bar(r2, df['Wins B'], width=bar_width, color=get_colors([combo[1] for combo in df.index]))
    style_axes(ax, 'Wins for Player Combos', 'Wins')
    annotate_bars(ax, bars1)
    annotate_bars(ax, bars2)
    ax.set_xticks([r + bar_width for r in range(len(df))])
    ax.set_xticklabels([f"{combo[0]} vs {combo[1]}" for combo in df.index], rotation=0, ha='right')
    plt.tight_layout()
    st.pyplot(fig, transparent=True)


def plot_player_combo_total_score_graph(df):
    # Bar width
    bar_width = 0.35

    # Wins Graph
    fig, ax = plt.subplots(figsize=(10, 6))
    r1 = np.arange(len(df))
    r2 = [x + bar_width for x in r1]
    
    # Total Scores Graph
    bars1 = ax.bar(r1, df['Total Score A'], width=bar_width, color=get_colors([combo[0] for combo in df.index]))
    bars2 = ax.bar(r2, df['Total Score B'], width=bar_width, color=get_colors([combo[1] for combo in df.index]))
    style_axes(ax, 'Total Scores for Player Combos', 'Total Score')
    annotate_bars(ax, bars1)
    annotate_bars(ax, bars2)
    ax.set_xticks([r + bar_width for r in range(len(df))])
    ax.set_xticklabels([f"{combo[0]} vs {combo[1]}" for combo in df.index], rotation=0, ha='right')
    plt.tight_layout()
    st.pyplot(fig, transparent=True)

def display_check_out_match_statistics():
    # Existing code for statistics:
    df_sheet = pd.read_csv(st.secrets["public_gsheets_url"])
    df_sheet["date"] = df_sheet["date"].astype(str)
    list_of_available_dates = list(set(df_sheet["date"].tolist()))
    selected_items = st.multiselect('Choose one or several matchday(s):', list_of_available_dates, placeholder="ask and thou ball receive")
    
    df_sheet['parsed_sheet_df'] = df_sheet.apply(lambda x: extract_data_from_games(x["games"], x["date"]), axis=1)
    df_tmp = pd.DataFrame()
    for _, row in df_sheet.iterrows():
        df_tmp = pd.concat([df_tmp, row["parsed_sheet_df"]])
    df = df_tmp[df_tmp["date"].isin(selected_items)].copy()

    if df.empty!=True:
        # Derive player and combination stats
        players_stats = calculate_player_stats(df)
        combination_stats = calculate_combination_stats(df)
        
        # Derive results
        results = derive_results(df)
        
        # Calculate win and loss streaks
        streaks = calculate_streaks(results)
        streaks = streaks.T.sort_values(["longest_win_streak", "longest_loss_streak"], ascending=False)         
            
        with st.expander("Individual player stats"):
            wins_and_total_scores_tab, wins_and_losses_tab, streaks_tab, trends_tab = st.tabs(["Wins and total scores", "Wins and losses over time", "Streaks", "Trends"])
            with wins_and_total_scores_tab:
                plot_wins_and_total_scores(players_stats)
            with wins_and_losses_tab:
                wins_and_losses_over_time_plot(results)
            with streaks_tab:
                graph_win_and_loss_streaks(streaks)
            with trends_tab:
                win_loss_trends_plot(results)
        with st.expander("Player combination stats"):
            wins_tab, scores_tab = st.tabs(["Wins", "Scores"])
            with wins_tab:
                plot_player_combo_wins_graph(combination_stats)
            with scores_tab:
                plot_player_combo_total_score_graph(combination_stats)

display_check_out_match_statistics()
