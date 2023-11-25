import gspread
from google.oauth2.service_account import Credentials
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import numpy as np
import streamlit as st
import re
import sys
import io
import itertools


class DualOutput:
    """A simple class to write outputs to two streams."""

    def __init__(self, stream1, stream2):
        self.stream1 = stream1
        self.stream2 = stream2

    def write(self, message):
        self.stream1.write(message)
        self.stream2.write(message)

    def flush(self):
        self.stream1.flush()
        self.stream2.flush()


def capture_stdout(func, *args, **kwargs):
    """Capture the stdout of a function and return its value and printed output."""
    original_stdout = sys.stdout
    buffer = io.StringIO()

    # Set stdout to write both to the buffer and the original stdout
    sys.stdout = DualOutput(original_stdout, buffer)

    value = func(*args, **kwargs)
    sys.stdout = original_stdout
    return value, buffer.getvalue()


def strip_ansi_escape_codes(s):
    """
    Remove ANSI escape codes from a string.
    """
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', s)


def write_prompt_response_to_database(prompt, response):
    spreadsheetId = st.secrets["public_gsheets_id"]
    scope = ['https://www.googleapis.com/auth/spreadsheets']
    google_creds_dict = dict(st.secrets["google_creds"])
    credentials = Credentials.from_service_account_info(google_creds_dict, scopes=scope)
    client = gspread.authorize(credentials)
    spreadsheet = client.open_by_key(spreadsheetId)
    worksheet = spreadsheet.worksheet("chat_history")
    worksheet.append_row([prompt, response])


def extract_data_from_games(games, date):
    pattern = r'([A-Za-z]+|S)\s-\s([A-Za-z]+|L)\s(\d{1,2}:\d{1,2})'
    matches = re.findall(pattern, games)

    processed_data = [[m[0], m[1], int(m[2].split(':')[0]), int(m[2].split(':')[1])] for m in matches]

    df = pd.DataFrame(processed_data, columns=["First Name", "Second Name", "First Score", "Second Score"])

    for i in ["First Name", "Second Name"]:
        df[i] = (
            np.where(df[i].str.startswith("S"), "Simon", 
                     np.where(df[i].str.startswith("F"), "Friedemann",
                              np.where(df[i].str.startswith("L"), "Lucas", 
                                       np.where(df[i].str.startswith("T"), "Tobias",
                                                np.where(df[i].str.startswith("P"), "Peter",
                                                         "unknown"))))))

    df["date"] = date
    return df


def get_name_opponent_name_df(df):
    # Create two new dataframes, one for the first players and one for the second players
    df_first = df[['index', 'First Name', 'First Score', 'Second Score', 'date', 'Second Name']].copy()
    df_second = df[['index', 'Second Name', 'Second Score', 'First Score', 'date', 'First Name']].copy()

    # Rename the columns
    df_first.columns = ['match_number', 'Name', 'Player Score', 'Opponent Score', 'Date', 'Opponent Name']
    df_second.columns = ['match_number', 'Name', 'Player Score', 'Opponent Score', 'Date', 'Opponent Name']

    # Add a new column indicating whether the player won or lost
    df_first['Wins'] = df_first['Player Score'] > df_first['Opponent Score']
    df_second['Wins'] = df_second['Player Score'] > df_second['Opponent Score']

    # Add a new column with the score difference
    df_first['Score Difference'] = df_first['Player Score'] - df_first['Opponent Score']
    df_second['Score Difference'] = df_second['Player Score'] - df_second['Opponent Score']

    # Concatenate the two dataframes
    df_new = pd.concat([df_first, df_second])
    # Sort the new dataframe by date
    df_new.sort_values('match_number', inplace=True)

    # Reset the index
    df_new.reset_index(drop=True, inplace=True)

    # Convert the Win column to numeric values
    df_new['WinsNumeric'] = df_new['Wins'].astype(int)

    # Calculate the cumulative sum of wins for each player
    df_new['CumulativeWins'] = df_new.groupby('Name')['WinsNumeric'].cumsum()

    # Calculate the cumulative sum of wins for each player
    df_new['CumulativeTotal Score'] = df_new.groupby('Name')['Player Score'].cumsum()

    # For each player, create a column that represents the number of the game for that player
    df_new['PlayerGameNumber'] = df_new.groupby('Name').cumcount() + 1

    return df_new


def calculate_combination_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate statistics based on player combinations.

    Args:
    - df (pd.DataFrame): Dataframe containing match results.

    Returns:
    - pd.DataFrame: A dataframe containing combination statistics.
    """
    df['Player Combo'] = df.apply(lambda x: tuple(sorted([x['First Name'], x['Second Name']])), axis=1)
    df['Score Combo'] = df.apply(
        lambda x: (x['First Score'], x['Second Score']) if x['First Name'] < x['Second Name'] else (
        x['Second Score'], x['First Score']), axis=1)
    df['Winner'] = df.apply(lambda x: x['First Name'] if x['First Score'] > x['Second Score'] else x['Second Name'],
                            axis=1)

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


def derive_results(df):
    results = {
        'Simon': [],
        'Friedemann': [],
        'Lucas': []
    }

    for _, row in df.iterrows():
        if row['Player Score'] > row['Opponent Score']:
            results[row['Name']].append(1)  # Win for First Name
        elif row['Player Score'] < row['Opponent Score']:
            results[row['Name']].append(-1)  # Loss for First Name
        else:
            results[row['Name']].append(0)  # Tie for First Name
    return results


def get_streak_counter(df):
    # Create a new column to detect a change in the 'Win' column
    df['change'] = df['Wins'].ne(df['Wins'].shift()).astype(int)

    # Group by the cumulative sum of the 'change' column to identify each streak and create a counter for each streak
    df['streak_counter'] = df.groupby(df['change'].cumsum()).cumcount() + 1
    return df


def get_name_streaks_df(df_new):
    df_streaks = pd.DataFrame()
    for name in ["Lucas", "Simon", "Friedemann"]:
        df_streak_tmp = df_new[df_new["Name"] == name].copy()
        df_streak_tmp = get_streak_counter(df_streak_tmp)
        longest_win_streak = df_streak_tmp[df_streak_tmp["Wins"] == True]["streak_counter"].max()
        longest_loss_streak = df_streak_tmp[df_streak_tmp["Wins"] == False]["streak_counter"].max()
        df_streak_name = pd.DataFrame(
            [{"Name": name, 'longest_win_streak': longest_win_streak, 'longest_loss_streak': longest_loss_streak}])
        df_streaks = pd.concat([df_streaks, df_streak_name])
    df_streaks = df_streaks.reset_index(drop=True).copy()
    return df_streaks


def win_loss_trends_plot(results, player_colors, title_color):
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
        # ax.grid(False)  # Turn off grid
        plt.tight_layout()
        st.pyplot(fig, transparent=True)


def wins_and_losses_over_time_plot(results, player_colors, title_color):
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
        # ax.grid(False)  # Turn off grid
        plt.tight_layout()
        st.pyplot(fig, transparent=True)


def plot_wins_and_total_scores(df2, title_color):
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


def graph_win_and_loss_streaks(df1, title_color):
    # Define colors for the win and loss streaks
    colors = {'longest_win_streak': 'green', 'longest_loss_streak': 'red'}

    ax = df1.set_index('Name')[['longest_win_streak', 'longest_loss_streak']].plot(kind='bar', figsize=(10, 6),
                                                                                   color=[colors[col] for col in
                                                                                          ['longest_win_streak',
                                                                                           'longest_loss_streak']])
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


def get_colors(players, color_map):
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


def plot_player_combo_graph(df, color_map, entity):
    # Bar width
    bar_width = 0.35

    # Wins Graph
    fig, ax = plt.subplots(figsize=(10, 6))
    r1 = np.arange(len(df))
    r2 = [x + bar_width for x in r1]

    # Total Scores Graph
    bars1 = ax.bar(r1, df[f'{entity} A'], width=bar_width,
                   color=get_colors([combo[0] for combo in df.index], color_map=color_map))
    bars2 = ax.bar(r2, df[f'{entity} B'], width=bar_width,
                   color=get_colors([combo[1] for combo in df.index], color_map=color_map))
    annotate_bars(ax, bars1)
    annotate_bars(ax, bars2)
    ax.set_xticks([r + bar_width for r in range(len(df))])
    ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6)
    ax.set_xticklabels([f"{combo[0]} vs {combo[1]}" for combo in df.index], rotation=0, ha='right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, transparent=True)


def plot_bars(df2, title_color, player_colors, entity):
    # bar_width = 0.35
    r1 = np.arange(len(df2[entity]))  # positions for Wins bars

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot 'Wins' - modified this part to iterate over players and use respective colors
    for idx, player in enumerate(df2.index):
        ax1.bar(r1[idx],
                df2[entity][idx],
                color=player_colors[player],
                alpha=1.0,
                # width=bar_width,
                label=player)

    ax1.set_ylabel(entity, color=title_color)  # Changed to title_color
    ax1.tick_params(axis='y', labelcolor=title_color)  # Changed to title_color
    ax1.tick_params(axis='x', colors=title_color)

    # Annotations for Wins
    for idx, player in enumerate(df2.index):
        yval = df2[entity][idx]
        ax1.text(r1[idx], yval + 0.5, int(yval), ha='center', va='bottom', color=title_color)

    # Adjust x-tick labels to center under the bars
    ax1.set_xticks(r1)
    ax1.set_xticklabels(df2.index, ha='center')

    # Grid and styling
    ax1.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6)
    # ax1.set_title(f'{entity} for Players', color=title_color)

    fig.patch.set_alpha(0.0)
    ax1.set_facecolor((0, 0, 0, 0))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig, transparent=True)


def cumulative_wins_over_time(df, color_map, title_color, entity):
    fig, ax1 = plt.subplots(figsize=(15, 7))

    # For each player, plot the cumulative sum of wins over their respective game number
    for name, group in df.groupby('Name'):
        ax1.plot(group['PlayerGameNumber'], group[f'Cumulative{entity}'], label=name, color=color_map[name], linewidth=5.0)

        # Annotate the last point with the player's name
        last_point = group.iloc[-1]
        ax1.text(last_point['PlayerGameNumber'] + 0.5, last_point[f'Cumulative{entity}'], name, color=color_map[name],
                 verticalalignment='center')

    ax1.set_title(f'Cumulative {entity} Over Time for Each Player', color=title_color)
    ax1.set_ylabel(f'Cumulative {entity}', color=title_color)
    ax1.set_xlabel('Player Game Number', color=title_color)
    ax1.tick_params(axis='y', labelcolor=title_color)
    ax1.tick_params(axis='x', colors=title_color)
    ax1.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6)

    # Making plot transparent and removing spines
    fig.patch.set_alpha(0.0)
    ax1.set_facecolor((0, 0, 0, 0))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)

    # Annotations for Cumulative Wins
    for idx, row in df.iterrows():
        ax1.text(row['PlayerGameNumber'], row[f'Cumulative{entity}'] + 0.2, int(row[f'Cumulative{entity}']), ha='center',
                 va='bottom', color=title_color)

    plt.tight_layout()
    st.pyplot(plt, transparent=True)


def entities_face_to_face_over_time(df, color_map, title_color, entity):
    # Getting unique player combinations
    players = df['Name'].unique()
    combinations = list(itertools.combinations(players, 2))

    for comb in combinations:
        fig, ax1 = plt.subplots(figsize=(15, 7))

        # Filtering dataframe for games where the two players from the combination played against each other
        matched_games = []
        for i in range(0, len(df) - 1, 2):
            if set(df.iloc[i:i + 2]['Name'].values) == set(comb):
                matched_games.extend([df.iloc[i], df.iloc[i + 1]])

        matched_df = pd.DataFrame(matched_games).reset_index(drop=True)

        # Get cumulative wins for each player within this filtered dataframe
        matched_df[f'Cumulative{entity}'] = matched_df.groupby('Name')[entity].cumsum()

        # Plotting the cumulative wins for each player with consistent colors
        for player in comb:
            player_data = matched_df[matched_df['Name'] == player]
            ax1.plot(player_data.index // 2 + 1, player_data[f'Cumulative{entity}'], color=color_map[player], linewidth=5.0)

            # Annotate the last point with the player's name
            last_point = player_data.iloc[-1]
            ax1.text(last_point.name // 2 + 1.2, last_point[f'Cumulative{entity}'], player, color=color_map[player],
                     verticalalignment='center')

        ax1.set_title(f'Cumulative {entity} Between {comb[0]} and {comb[1]}', color=title_color)
        ax1.set_ylabel(f'Cumulative {entity} Against Each Other', color=title_color)
        ax1.set_xlabel('Game Number Between The Two', color=title_color)
        ax1.tick_params(axis='y', labelcolor=title_color)
        ax1.tick_params(axis='x', colors=title_color)
        ax1.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6)

        # Making plot transparent and removing spines
        fig.patch.set_alpha(0.0)
        ax1.set_facecolor((0, 0, 0, 0))
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)

        # Annotations
        for idx, row in matched_df.iterrows():
            yval = row[f'Cumulative{entity}']
            ax1.text(idx // 2 + 1, yval + 0.2, int(yval), ha='center', va='bottom', color=title_color)

        plt.tight_layout()
        st.pyplot(plt, transparent=True)
