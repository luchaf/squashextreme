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
from fuzzywuzzy import process

def correct_name(name, name_list, threshold=85):
    """
    Corrects the name to the most similar one in the provided list if similarity is above the threshold.
    :param name: The name to be corrected.
    :param name_list: List of correct names.
    :param threshold: The minimum similarity score to consider a match.
    :return: Corrected name.
    """
    highest_match = process.extractOne(name, name_list, score_cutoff=threshold)
    return highest_match[0] if highest_match else name

def correct_names_in_dataframe(df, columns, name_list):
    """
    Corrects names in specified DataFrame columns.
    :param df: The DataFrame containing names.
    :param columns: List of columns to be corrected.
    :param name_list: List of correct names.
    :return: DataFrame with corrected names.
    """
    for column in columns:
        df[column] = df[column].apply(lambda name: correct_name(name, name_list))
    return df


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

def wins_and_losses_over_time_plot(results, player_colors, title_color):
    # Iterate through each player in the results
    for player, res in results.items():
        # Create the figure with Plotly
        fig = go.Figure()

        # Adding the line chart for wins and losses
        fig.add_trace(go.Scatter(
            x=list(range(len(res))),  # Assuming the index represents games played
            y=res,
            mode='lines+markers',
            name=player,
            line=dict(color=player_colors[player]),
            marker=dict(symbol='circle')
        ))

        # Update the layout for aesthetics and labels
        fig.update_layout(
            title=dict(text=f"Wins and Losses for {player} Over Time", x=0.5, xanchor='center', font=dict(size=16, color=title_color)),
            xaxis=dict(title='Games', titlefont=dict(size=14, color=title_color), tickcolor=title_color),
            yaxis=dict(title='Result', titlefont=dict(size=14, color=title_color), tickvals=[-1, 1], ticktext=['Loss', 'Win'], tickcolor=title_color),
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis_zeroline=False,  # Hide zero line
            xaxis_zeroline=False,
            showlegend=False
        )

        # Update axes properties
        fig.update_xaxes(showline=True, linewidth=2, linecolor=title_color, mirror=True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor=title_color, mirror=True)

        # Display the figure using Streamlit
        st.plotly_chart(fig, use_container_width=True)

def wins_and_losses_over_time_plot(results, player_colors, title_color):
    # Iterate through each player in the results
    for player, res in results.items():
        # Create the figure with Plotly
        fig = go.Figure()

        # Adding the line chart for wins and losses
        fig.add_trace(go.Scatter(
            x=list(range(len(res))),
            y=res,
            mode='lines+markers',
            name=player,
            line=dict(color=player_colors[player]),
            marker=dict(symbol='circle')
        ))

        # Update the layout for aesthetics and labels
        fig.update_layout(
            title=dict(text=f"Wins and Losses Over Time", x=0.5, xanchor='center', font=dict(size=16, color=title_color)),
            xaxis=dict(title='Games', titlefont=dict(size=14, color=title_color), tickcolor=title_color),
            yaxis=dict(title='Result', titlefont=dict(size=14, color=title_color), tickvals=[-1, 1], ticktext=['Loss', 'Win'], tickcolor=title_color),
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis_zeroline=False,  # Hide zero line
            xaxis_zeroline=False,
            showlegend=False
        )

        # Adding an annotation for the player's name
        fig.add_annotation(
            text=player,  # Player's name
            xref="paper", yref="paper",
            x=0.5, y=1.07,  # Adjust these positions as needed
            showarrow=False,
            font=dict(size=20, color=player_colors[player]),  # Adjust font size and color as needed
            align="center",
            xanchor='center',
            yanchor='bottom'
        )

        # Update axes properties
        fig.update_xaxes(showline=True, linewidth=2, linecolor=title_color, mirror=True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor=title_color, mirror=True)

        # Display the figure using Streamlit
        st.plotly_chart(fig, use_container_width=True)



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

    # Prepare data for Plotly
    data = [
        go.Bar(
            name='Longest Win Streak',
            x=df1['Name'],
            y=df1['longest_win_streak'],
            marker=dict(color=colors['longest_win_streak'])
        ),
        go.Bar(
            name='Longest Loss Streak',
            x=df1['Name'],
            y=df1['longest_loss_streak'],
            marker=dict(color=colors['longest_loss_streak'])
        )
    ]

    # Create the figure with the data
    fig = go.Figure(data=data)

    # Update layout for aesthetics and labels, removing the title
    fig.update_layout(
        yaxis=dict(title='Number of Matches', fixedrange=True, titlefont=dict(size=14, color=title_color)),
        xaxis=dict(title='Players', fixedrange=True ,titlefont=dict(size=14, color=title_color), title_standoff=10),
        plot_bgcolor='rgba(0,0,0,0)', # Fully transparent background for the plot
        paper_bgcolor='rgba(0,0,0,0)', # Fully transparent background for the paper
        font=dict(color=title_color),
        margin=dict(l=10, r=10, t=10, b=10), # Adjust margins if necessary
        bargap=0.2, # Adjust the spacing between bars
        showlegend=True,
                legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.5,  # Adjust this value as needed to place the legend under the X-axis title
            xanchor='center',
            x=0.5
        ),
    )

    # Streamlit Plotly display
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def get_colors(players, color_map):
    return [color_map[player] for player in players]

# Function to plot the interactive bar chart
def plot_player_combo_graph(df, color_map, entity, relative=False):
    st.title(f'{entity} Comparison per Player Combination')

    # Get a list of all players involved
    all_players = sorted(set(idx for idx_pair in df.index for idx in idx_pair))

    # Generate a unique key for the multiselect widget based on the entity
    unique_key = f'player_select_{entity}_{relative}'
    
    # Use Streamlit's multiselect widget to allow selection of multiple players
    selected_players = st.multiselect(
        'Select players:', all_players, default=all_players, key=unique_key
    )

    # Initialize the figure
    fig = go.Figure()

    # Loop through each player combination and add bars to the figure
    for idx, row in df.iterrows():
        player_a, player_b = idx
        if player_a in selected_players and player_b in selected_players:
            value_a = row[f'{entity} A']
            value_b = row[f'{entity} B']
            total = value_a + value_b

            if relative:
                # Calculate percentages if relative is True
                bottom_value = (value_a / total) * 100 if total > 0 else 0
                top_value = (value_b / total) * 100 if total > 0 else 0
            else:
                # Use absolute values if relative is False
                bottom_value = value_a
                top_value = value_b

            # Determine which player should be at the bottom or top based on the values
            bottom_player, top_player = (player_a, player_b) if bottom_value > top_value else (player_b, player_a)

            # Swap values if necessary
            if bottom_player != player_a:
                bottom_value, top_value = top_value, bottom_value

            # Adding a trace for the bottom player
            fig.add_trace(go.Bar(
                x=[f'{player_a} vs {player_b}'],
                y=[bottom_value],
                name=bottom_player,
                marker=dict(color=color_map.get(bottom_player, '#333')),
                hoverinfo='y+text',
                hovertext=[f'{entity} for {bottom_player}: {bottom_value:.2f}' + ('%' if relative else '')],
            ))
            # Adding a trace for the top player
            fig.add_trace(go.Bar(
                x=[f'{player_a} vs {player_b}'],
                y=[top_value],
                name=top_player,
                marker=dict(color=color_map.get(top_player, '#333')),
                hoverinfo='y+text',
                hovertext=[f'{entity} for {top_player}: {top_value:.2f}' + ('%' if relative else '')],
                showlegend=False
            ))

    # Set up the figure layout for a stacked bar chart
    y_axis_title = f'{entity} Percentage' if relative else f'{entity} Scores'
    y_axis_ticksuffix = '%' if relative else ''
    fig.update_layout(
        barmode='stack',
        title=f'{("Relative" if relative else "Absolute")} {entity} Comparison per Player Combination',
        xaxis=dict(title='Player Combinations', fixedrange=True),
        yaxis=dict(title=y_axis_title, ticksuffix=y_axis_ticksuffix, fixedrange=True),
        hovermode='x',
        showlegend=False
    )

    # Show the figure
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


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
        yaxis=dict(title=entity, fixedrange=True),
        xaxis=dict(title='Players', tickangle=-45, fixedrange=True, title_standoff=25),  # Increased standoff for clarity
        plot_bgcolor='rgba(0,0,0,0)',  # Fully transparent background for the plot
        paper_bgcolor='rgba(0,0,0,0)',  # Fully transparent background for the paper
        margin=dict(l=10, r=10, t=10, b=80),  # Increased bottom margin to accommodate legend
        bargap=0.2,  # Adjust the spacing between bars
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.5,  # Adjust this value as needed to place the legend under the X-axis title
            xanchor='center',
            x=0.5
        )
    )

    # When using Streamlit to display the chart
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


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
            line=dict(color=color_map[name], width=2),
            marker=dict(size=3),  # Adjust marker size as needed
        ))

    # Update the layout for the figure
    fig.update_layout(
        title=f'Cumulative {entity} Over Time for Each Player',
        xaxis=dict(title='Player Game Number', color=title_color, fixedrange=True),
        yaxis=dict(title=f'Cumulative {entity}', color=title_color, fixedrange=True),
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        font=dict(color=title_color),
        hovermode='closest',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.5,  # Adjust this value as needed to place the legend under the X-axis title
            xanchor='center',
            x=0.5
        )
    )

    # Display the interactive plot
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def cumulative_win_ratio_over_time(df, color_map, title_color):
    # Initialize a Plotly figure
    fig = go.Figure()

    # For each player, calculate the cumulative wins and win ratio over their respective game number
    for name, group in df.groupby('Name'):
        # Sort the group by game number to ensure the cumulative sum is correct
        group = group.sort_values('PlayerGameNumber')
        # Calculate cumulative wins
        group['CumulativeWins'] = group['Wins'].cumsum()
        # Calculate the cumulative win ratio
        group['CumulativeWinRatio'] = group['CumulativeWins'] / group['PlayerGameNumber']

        fig.add_trace(go.Scatter(
            x=group['PlayerGameNumber'],
            y=group['CumulativeWinRatio'],
            mode='lines+markers',
            name=name,
            line=dict(color=color_map[name], width=2),
            marker=dict(size=3),
        ))

    # Update the layout for the figure
    fig.update_layout(
        title='Cumulative Win Ratio Over Time for Each Player',
        xaxis=dict(title='Player Game Number', color=title_color, fixedrange=True),
        yaxis=dict(title='Cumulative Win Ratio', color=title_color, fixedrange=True),
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        font=dict(color=title_color),
        hovermode='closest',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.5,  # Adjust this value as needed to place the legend under the X-axis title
            xanchor='center',
            x=0.5
        )
    )

    # Display the interactive plot
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def cumulative_win_ratio_over_time(df, color_map, title_color):
    # Initialize a Plotly figure
    fig = go.Figure()

    # For each player, calculate the cumulative wins and win ratio over their respective game number
    for name, group in df.groupby('Name'):
        # Sort the group by game number to ensure the cumulative sum is correct
        group = group.sort_values('PlayerGameNumber')
        # Calculate cumulative wins
        group['CumulativeWins'] = group['Wins'].cumsum()
        # Calculate the cumulative win ratio
        group['CumulativeWinRatio'] = group['CumulativeWins'] / group['PlayerGameNumber']
        # Calculate the median win ratio
        median_win_ratio = group['CumulativeWinRatio'].median()

        fig.add_trace(go.Scatter(
            x=group['PlayerGameNumber'],
            y=group['CumulativeWinRatio'],
            mode='lines+markers',
            name=name,
            line=dict(color=color_map[name], width=2),
            marker=dict(size=3),
        ))

        # Add a horizontal line for the median win ratio
        fig.add_trace(go.Scatter(
            x=[group['PlayerGameNumber'].min(), group['PlayerGameNumber'].max()],
            y=[median_win_ratio, median_win_ratio],
            mode='lines',
            name=f'Median {name}',
            line=dict(color=color_map[name], width=2, dash='dash'),
            hoverinfo='y+name',  # Show only the y value and the trace name on hover
            text=f'Median Win Ratio: {median_win_ratio:.2f}',  # Text to display on hover, formatted to 2 decimal places
        ))

    # Update the layout for the figure
    fig.update_layout(
        title='Cumulative Win Ratio Over Time for Each Player',
        xaxis=dict(title='Player Game Number', color=title_color, fixedrange=True),
        yaxis=dict(title='Cumulative Win Ratio', color=title_color, fixedrange=True),
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        font=dict(color=title_color),
        hovermode='closest',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.5,  # Adjust this value as needed to place the legend under the X-axis title
            xanchor='center',
            x=0.5
        )
    )

    # Display the interactive plot
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})




def entities_face_to_face_over_time_abs(df, color_map, title_color, entity):
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
                    line=dict(color=color_map[player], width=2),
                    marker=dict(size=3),
                    showlegend=True
                ))

            # Update the layout for each figure
            fig.update_layout(
                title=f'Cumulative {entity} Between {comb[0]} and {comb[1]}',
                xaxis=dict(title='Game Number Between The Two', color=title_color, fixedrange=True),
                yaxis=dict(title=f'Cumulative {entity}', color=title_color, fixedrange=True),
                hovermode='closest',
                        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.5,  # Adjust this value as needed to place the legend under the X-axis title
            xanchor='center',
            x=0.5
        ),
            )

            # Display the plot for the current combination
            st.plotly_chart(fig, use_container_width=True,config={'displayModeBar': False})


def entities_face_to_face_over_time_rel(df, color_map, title_color, entity):
    players = df['Name'].unique()
    all_combinations = list(itertools.combinations(players, 2))

    for comb in all_combinations:
        matched_games = []
        for i in range(0, len(df) - 1, 2):
            if set(df.iloc[i:i + 2]['Name']) == set(comb):
                matched_games.extend([df.iloc[i], df.iloc[i + 1]])

        matched_df = pd.DataFrame(matched_games).reset_index(drop=True)

        if not matched_df.empty:
            fig = go.Figure()

            for player in comb:
                player_data = matched_df[matched_df['Name'] == player]
                total_games = player_data.index // 2 + 1
                cumulative_wins = player_data[entity].cumsum()
                win_ratio = cumulative_wins / total_games

                fig.add_trace(go.Scatter(
                    x=total_games,
                    y=win_ratio,
                    mode='lines+markers',
                    name=player,
                    line=dict(color=color_map[player], width=2),
                    marker=dict(size=3),
                    showlegend=True
                ))

            fig.update_layout(
                title=f'Cumulative Win Ratio Between {comb[0]} and {comb[1]}',
                xaxis=dict(title='Game Number Between The Two', color=title_color, fixedrange=True),
                yaxis=dict(title=f'Cumulative Win Ratio', color=title_color, fixedrange=True),
                hovermode='closest',
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=-0.5,
                    xanchor='center',
                    x=0.5
                ),
            )

            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


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
                line=dict(color=color_map[combination[0]], width=2),
                marker=dict(size=3),
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
                xaxis=dict(title='Match Number', color=title_color, fixedrange=True),
                yaxis=dict(title='Score Difference (Vorsprung)', color=title_color, fixedrange=True),
                legend_title=dict(text='Players'),
                hovermode='closest',
                        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.5,  # Adjust this value as needed to place the legend under the X-axis title
            xanchor='center',
            x=0.5
        ),
            )

            # Display the plot for the current combination
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


