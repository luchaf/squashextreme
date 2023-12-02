from utils import (extract_data_from_games, get_name_opponent_name_df, get_name_streaks_df,
                   calculate_combination_stats,
                   derive_results,
                   win_loss_trends_plot,
                   wins_and_losses_over_time_plot,
                   graph_win_and_loss_streaks,
                   plot_player_combo_graph,
                   plot_bars, cumulative_wins_over_time, entities_face_to_face_over_time, closeness_of_matches_over_time)
import streamlit as st
import pandas as pd
from datetime import date

import matplotlib.pyplot as plt

# Streamlit app
st.title('Overanalysis Oasis')

# Load data from database
df = pd.DataFrame()
df_sheet = pd.read_csv(st.secrets["public_gsheets_url"])
df_sheet["date"] = df_sheet["date"].astype(str)
list_of_available_dates = list(set(df_sheet["date"].tolist()))
df_sheet['parsed_sheet_df'] = df_sheet.apply(lambda x: extract_data_from_games(x["games"], x["date"]), axis=1)
df_tmp = pd.DataFrame()
for _, row in df_sheet.iterrows():
    df_tmp = pd.concat([df_tmp, row["parsed_sheet_df"]])

(
    settings_tab,
    basic_metrics_tab,
    intermediate_metrics_tab,
) = st.tabs(["Settings :gear:",
             "Basic Metrics :star:",
             "Intermediate Metrics :exploding_head:",
             ])

with settings_tab:
    with (st.expander("Lob your preferred time period into the analysis court.")):
        # Sample data: list of dates when matches occurred
        match_dates = list(set(df_sheet["date"].tolist()))

        all_match_days = st.checkbox('All-court days', value=True)
        specific_match_day = st.checkbox('That one day on the court')
        date_range = st.checkbox('Court calendar slice')

        if specific_match_day:
            if all_match_days or date_range:
                # Warn the user if they select multiple options
                st.warning('Please select only one option.')
                st.stop()

            sorted_dates = sorted(match_dates)
            dates_str = ", ".join([d for d in sorted_dates])
            st.info(f"Available match dates: {dates_str}", icon="ℹ️")

            match_day = st.date_input('Select a specific match day')
            match_day = match_day.strftime('%Y%m%d')
            if match_day in match_dates:
                st.write(f"You've selected {match_day} as the match day of interest!")
                df = df_tmp[df_tmp["date"] == match_day].copy()
            else:
                st.warning(f"No matches on {match_day}. Please refer to the list for match days.")

        elif all_match_days:
            if date_range:
                # Warn the user if they select multiple options
                st.warning('Please select only one option.')
                st.stop()

            st.write("You've selected analysis for all match days!")
            df = df_tmp.copy()

        elif date_range:
            start_date = st.date_input('Start Date', value=date.today())
            start_date = start_date.strftime('%Y%m%d')
            end_date = st.date_input('End Date', value=date.today())
            end_date = end_date.strftime('%Y%m%d')

            if start_date > end_date:
                st.warning('Start date should be before or the same as end date.')
            else:
                matches_in_range = [d for d in match_dates if start_date <= d <= end_date]
                if matches_in_range:
                    sorted_dates = sorted(matches_in_range)
                    dates_str = ", ".join([d for d in sorted_dates])
                    st.info(f"Matches between {start_date} and {end_date}: {dates_str}", icon="ℹ️")
                    df = df_tmp[((df_tmp["date"] >= start_date) & (df_tmp["date"] <= end_date))].copy()
                else:
                    st.warning(f"No matches between {start_date} and {end_date}.")

    with st.expander("Adjust aesthetics"):
        col_friede, col_simon, col_lucas, col_peter, col_tobias = st.columns(5)
        with col_friede:
            color_friedemann = st.color_picker('Friedemann', '#ffc0cb')
        with col_simon:
            color_simon = st.color_picker('Simon', '#004d9d')
        with col_lucas:
            color_lucas = st.color_picker('Lucas', '#7CFC00')
        with col_peter:
            color_peter = st.color_picker('Peter', '#FCBA20')
        with col_tobias:
            color_tobias = st.color_picker('Tobias', '#00FCF8')
        player_colors = {
            'Simon': color_simon,
            'Friedemann': color_friedemann,
            'Lucas': color_lucas,
          'Peter': color_peter,
          'Tobias': color_tobias,          
        }
        #title_color = 'black'
        # A color that works on both dark and light backgrounds
        title_color = '#CCCCCC'

    if df.empty:
        st.warning('Please select at least one valid matchday.')
    else:
        # Dominance Scores
        df = df.reset_index(drop=True).copy()
        df = df.reset_index()
        
        # Derive player and combination stats
        combination_stats = calculate_combination_stats(df)
        df = get_name_opponent_name_df(df)

        # Calculate individual stats
        players_stats = df.groupby('Name').agg({'Wins': 'sum', 'Player Score': 'sum'}).rename(
            columns={'Player Score': 'Total Score'}).sort_values("Wins", ascending=False)

        # Derive results
        results = derive_results(df)

        # Calculate win and loss streaks
        # streaks = calculate_streaks(results)
        streaks = get_name_streaks_df(df)
        streaks = streaks.sort_values(["longest_win_streak", "longest_loss_streak"], ascending=False)

        with basic_metrics_tab:
            Number_of_Wins_tab, Total_Points_Scored_tab = st.tabs(["# Wins", "# Points Scored"])
            with Number_of_Wins_tab:
                #st.info(f"How many wins did each player collect...", icon="❓")
                total_wins_tab, face_to_face_wins_tab = st.tabs(["Total", "Face-to-Face-Feud"])
                with total_wins_tab:
                    #st.info(f"...in total: static or over time", icon="❓")
                    wins_all_time_tab, wins_over_time_tab = st.tabs(["static", "over time"])
                    with wins_all_time_tab:
                        plot_bars(players_stats, title_color, player_colors, "Wins")
                    with wins_over_time_tab:
                        cumulative_wins_over_time(df, player_colors, title_color, "Wins")
                with face_to_face_wins_tab:
                    #st.info(f"...against specific opponents: static or over time", icon="❓")
                    wins_face_to_face_all_time_tab, wins_face_to_face_over_time_tab = st.tabs(["static", "over time"])
                    with wins_face_to_face_all_time_tab:
                        plot_player_combo_graph(combination_stats, player_colors, "Wins")
                    with wins_face_to_face_over_time_tab:
                        #plot_player_combo_graph(combination_stats, player_colors, "Wins")
                        entities_face_to_face_over_time(df, player_colors, title_color, "Wins")

            with Total_Points_Scored_tab:
                #st.info(f"How many points did each player score...", icon="❓")
                total_score_tab, face_to_face_score_tab = st.tabs(["Total", "Face-to-Face-Feud"])
                with total_score_tab:
                    #st.info(f"...in total: static or over time", icon="❓")
                    scores_all_time_tab, scores_over_time_tab = st.tabs(["static", "over time"])
                    with scores_all_time_tab:
                        plot_bars(players_stats, title_color, player_colors, "Total Score")
                    with scores_over_time_tab:
                        cumulative_wins_over_time(df, player_colors, title_color, "Total Score")
                with face_to_face_score_tab:
                    #st.info(f"...against specific opponents: static or over time", icon="❓")
                    scores_face_to_face_all_time_tab, scores_face_to_face_over_time_tab, competitiveness_tab = st.tabs(["static", "points over time", "competitiveness over time"])
                    with scores_face_to_face_all_time_tab:
                        plot_player_combo_graph(combination_stats, player_colors, "Total Score")
                    with scores_face_to_face_over_time_tab:
                        entities_face_to_face_over_time(df, player_colors, title_color, "Player Score")
                    with competitiveness_tab:
                        closeness_of_matches_over_time(df, player_colors, title_color)

        with intermediate_metrics_tab:
            (streaks_tab,
             trends_tab) = st.tabs(["Streaks",
                                    "Trends"])
            with streaks_tab:
                st.info(f"Longest number of consecutive wins or losses by each player", icon="❓")
                static_streaks, over_time_streaks = st.tabs(["Static", "Over Time"])
                with static_streaks:
                    graph_win_and_loss_streaks(streaks, title_color)
                with over_time_streaks:
                    wins_and_losses_over_time_plot(results, player_colors, title_color)
            with trends_tab:
                win_loss_trends_plot(results, player_colors, title_color)

