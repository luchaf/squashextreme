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
import plotly.graph_objs as go
import time


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

    # Convert the Win and Player Score column to numeric values
    df_new['WinsNumeric'] = df_new['Wins'].astype(int)
    df_new['Player Score'] = df_new['Player Score'].astype(int)

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
        'Lucas': [],
        'Tobias': [],
        'Peter': [],
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
    for name in ["Lucas", "Simon", "Friedemann", "Peter", "Tobias"]:
        df_streak_tmp = df_new[df_new["Name"] == name].copy()
        df_streak_tmp = get_streak_counter(df_streak_tmp)
        longest_win_streak = df_streak_tmp[df_streak_tmp["Wins"] == True]["streak_counter"].max()
        longest_loss_streak = df_streak_tmp[df_streak_tmp["Wins"] == False]["streak_counter"].max()
        df_streak_name = pd.DataFrame(
            [{"Name": name, 'longest_win_streak': longest_win_streak, 'longest_loss_streak': longest_loss_streak}])
        df_streaks = pd.concat([df_streaks, df_streak_name])
    df_streaks = df_streaks.fillna(0).copy()
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

# Function to plot the interactive bar chart
def plot_player_combo_graph(df, color_map, entity):
    st.title(f'{entity} Comparison per Player Combination')

    # Get a list of all players involved
    all_players = sorted(set(idx for idx_pair in df.index for idx in idx_pair))

    # Generate a unique key for the multiselect widget based on the entity
    unique_key = f'player_select_{entity}'
    
    # Use Streamlit's multiselect widget to allow selection of multiple players
    selected_players = st.multiselect(
        'Select players:', all_players, default=all_players, key=unique_key
    )

    # Initialize the figure
    fig = go.Figure()

    # Adjust the width of the bars and their opacity
    # Define bar width relative to the number of bars
    num_bars = len(df.index) * 2  # times 2 for pairs of bars per group
    bar_width = max(0.35, min(4, 1 / num_bars))  # Adjust the 0.4 for maximum bar width as needed
    bar_opacity = 0.6  # Set between 0 and 1 to make bars semi-transparent

    # Add bars for each player combination
    for idx, row in df.iterrows():
        player_a, player_b = idx
        if player_a in selected_players and player_b in selected_players:
            # Adding a trace for player A
            fig.add_trace(go.Bar(
                x=[f'{player_a} vs {player_b}'],
                y=[row[f'{entity} A']],
                name=player_a,
                marker=dict(color=color_map.get(player_a, '#333')),
                hoverinfo='y+text',
                hovertext=[f'{entity} for {player_a}'],
                width=bar_width,
                opacity=bar_opacity
            ))
            # Adding a trace for player B
            fig.add_trace(go.Bar(
                x=[f'{player_a} vs {player_b}'],
                y=[row[f'{entity} B']],
                name=player_b,
                marker=dict(color=color_map.get(player_b, '#333')),
                hoverinfo='y+text',
                hovertext=[f'{entity} for {player_b}'],
                width=bar_width,
                opacity=bar_opacity,
                showlegend=False
            ))

    # Set up the figure layout, adjusting the bargap if necessary
    fig.update_layout(
        barmode='group',
        bargap=0.15,  # Adjust the gap between bars of adjacent x-ticks
        bargroupgap=0.15, 
        title=f'{entity} Comparison per Player Combination',
        xaxis=dict(title='Player Combinations'),
        yaxis=dict(title=f'{entity} Scores'),
        hovermode='closest',
        showlegend=False,  # Hiding the legend as the selection is done through multiselect
    )

    # Show the figure
    st.plotly_chart(fig)



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

def plot_bars(df2, title_color, player_colors, entity):
    # Prepare data for Plotly
    data = []
    for idx, player in enumerate(df2.index):
        data.append(go.Bar(
            x=[player],
            y=[df2[entity][idx]],
            marker=dict(color=player_colors[player]),
            name=player
        ))

    # Create the figure with the data
    fig = go.Figure(data=data)

    # Update layout for aesthetics and labels
    fig.update_layout(
        yaxis=dict(title=entity, titlefont=dict(color=title_color)),
        xaxis=dict(title='Players', titlefont=dict(color=title_color), tickangle=-45),
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background for the plot
        paper_bgcolor='rgba(0,0,0,0.5)',  # Semi-transparent dark background for the figure
        font=dict(color=title_color),
        margin=dict(l=10, r=10, t=10, b=10),
        bargap=0.2,  # Adjust the spacing between bars
    )

    # Annotations for each bar with the title_color for high contrast
    for idx, player in enumerate(df2.index):
        fig.add_annotation(
            x=player,
            y=df2[entity][idx],
            text=str(int(df2[entity][idx])),
            font=dict(color=title_color),
            showarrow=False,
            yshift=10
        )

    # Streamlit Plotly display
    st.plotly_chart(fig, use_container_width=True)



def cumulative_wins_over_time(df, color_map, title_color, entity):
    # Initialize a Plotly figure
    fig = go.Figure()

    # For each player, plot the cumulative sum of wins over their respective game number
    for name, group in df.groupby('Name'):
        fig.add_trace(go.Scatter(
            x=group['PlayerGameNumber'],
            y=group[f'Cumulative{entity}'],
            mode='lines+markers',  # Only lines and markers, no text
            name=name,
            line=dict(color=color_map[name], width=4),
            marker=dict(size=8),  # Adjust marker size as needed
        ))

    # Update the layout for the figure
    fig.update_layout(
        title=f'Cumulative {entity} Over Time for Each Player',
        xaxis=dict(title='Player Game Number', color=title_color),
        yaxis=dict(title=f'Cumulative {entity}', color=title_color),
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        font=dict(color=title_color),
        hovermode='closest',
        legend_title=dict(text='Players'),
        showlegend=True
    )

    # Display the interactive plot
    st.plotly_chart(fig)


def entities_face_to_face_over_time(df, color_map, title_color, entity):
    # Getting unique player combinations
    players = df['Name'].unique()
    all_combinations = list(itertools.combinations(players, 2))
    
    # Loop over selected combinations to create a plot for each
    for comb in all_combinations:
        # Filtering dataframe for games where the two players from the combination played against each other
        matched_games = []
        for i in range(0, len(df) - 1, 2):
            if set(df.iloc[i:i + 2]['Name']) == set(comb):
                matched_games.extend([df.iloc[i], df.iloc[i + 1]])

        matched_df = pd.DataFrame(matched_games).reset_index(drop=True)

        if not matched_df.empty:
            # Initialize a Plotly figure for each combination
            fig = go.Figure()

            # Get cumulative wins for each player within this filtered dataframe
            matched_df[f'Cumulative{entity}'] = matched_df.groupby('Name')[entity].cumsum()

            # Plotting the cumulative wins for each player
            for player in comb:
                player_data = matched_df[matched_df['Name'] == player]
                fig.add_trace(go.Scatter(
                    x=player_data.index // 2 + 1,
                    y=player_data[f'Cumulative{entity}'],
                    mode='lines+markers',
                    name=player,
                    line=dict(color=color_map[player], width=4),
                    marker=dict(size=8),
                    showlegend=True
                ))

            # Update the layout for each figure
            fig.update_layout(
                title=f'Cumulative {entity} Between {comb[0]} and {comb[1]}',
                xaxis=dict(title='Game Number Between The Two', color=title_color),
                yaxis=dict(title=f'Cumulative {entity}', color=title_color),
                legend_title=dict(text='Players'),
                hovermode='closest'
            )

            # Display the plot for the current combination
            st.plotly_chart(fig)


def closeness_of_matches_over_time(df, color_map, title_color, future_matches=5):
    # Get unique player combinations
    player_combinations = df[['Name', 'Opponent Name']].values.tolist()
    unique_combinations = set(tuple(sorted(comb)) for comb in player_combinations)
    
    # Loop over unique player combinations to create a plot for each
    for combination in unique_combinations:
        combination_df = df[
            ((df['Name'] == combination[0]) & (df['Opponent Name'] == combination[1]))
        ]
        
        if not combination_df.empty:
            # Initialize a Plotly figure for each combination
            fig = go.Figure()
            
            # Plotting the score difference for each match within this combination
            match_numbers = list(range(1, len(combination_df) + 1))
            fig.add_trace(go.Scatter(
                x=match_numbers,
                y=combination_df['Score Difference'],
                mode='lines+markers',
                name=f'{combination[0]} vs {combination[1]}',
                line=dict(color=color_map[combination[0]], width=4),
                marker=dict(size=8),
                showlegend=True
            ))
            
            # Calculate the trendline data points
            trendline_x = list(range(1, len(match_numbers) + future_matches + 1))
            trendline_y = combination_df['Score Difference'].rolling(window=5).mean().tolist()
            
            # Extend the trendline data for future matches
            last_trendline_value = trendline_y[-1]
            for i in range(future_matches):
                trendline_x.append(len(match_numbers) + i + 1)
                trendline_y.append(last_trendline_value)
            
            # Add the extrapolated trendline to the graph
            fig.add_trace(go.Scatter(
                x=trendline_x,
                y=trendline_y,
                mode='lines',
                name=f'Trendline ({combination[0]} vs {combination[1]})',
                line=dict(color=color_map[combination[0]], width=2, dash='dash'),
                showlegend=True
            ))
            
            # Add a horizontal dashed black line at 0
            fig.update_layout(
                shapes=[dict(
                    type='line',
                    x0=1,
                    x1=len(match_numbers) + future_matches,
                    y0=0,
                    y1=0,
                    line=dict(color='black', width=2, dash='dash')
                )]
            )
            
            # Update the layout for each figure
            fig.update_layout(
                title=f'Closeness of Matches Over Time Between {combination[0]} and {combination[1]}',
                xaxis=dict(title='Match Number', color=title_color),
                yaxis=dict(title='Score Difference (Vorsprung)', color=title_color),
                legend_title=dict(text='Players'),
                hovermode='closest'
            )

            # Display the plot for the current combination
            st.plotly_chart(fig)

