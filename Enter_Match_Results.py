import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

def reset_session_state():
    """Helper function to reset session state."""
    st.session_state['player1_name'] = ''
    st.session_state['player1_score'] = None
    st.session_state['player2_name'] = ''
    st.session_state['player2_score'] = None
    st.session_state['matchday_input'] = None
    st.session_state['show_confirm'] = False
    st.session_state['data_written'] = False
    
def display_enter_match_results():
    # Initialize session state values if not already set
    if 'player1_name' not in st.session_state:
        st.session_state['player1_name'] = ''
    if 'player1_score' not in st.session_state:
        st.session_state['player1_score'] = None
    if 'player2_name' not in st.session_state:
        st.session_state['player2_name'] = ''
    if 'player2_score' not in st.session_state:
        st.session_state['player2_score'] = None
    if 'matchday_input' not in st.session_state:
        st.session_state['matchday_input'] = None
    if 'show_confirm' not in st.session_state:
        st.session_state['show_confirm'] = False
    if 'data_written' not in st.session_state:
        st.session_state['data_written'] = False

    if st.session_state['data_written']:
        st.success("Successfully wrote match result to database. Do you want to enter a new match result?")
        if st.button("Enter New Match Result"):
            reset_session_state()
            st.experimental_rerun()

    else:
        st.title("Match Score Input")
    
        # Use session state for values
        st.session_state['player1_name'] = st.text_input("Player 1 Name", st.session_state['player1_name'])
    
        if st.session_state['player1_name']:
            st.session_state['player1_score'] = st.number_input("Player 1 Score", min_value=0, value=st.session_state['player1_score'], step=1)
            if st.session_state['player1_score'] or st.session_state['player1_score'] == 0:
                st.session_state['player2_name'] = st.text_input("Player 2 Name", st.session_state['player2_name'])
                if st.session_state['player2_name']:
                    st.session_state['player2_score'] = st.number_input("Player 2 Score", min_value=0, value=st.session_state['player2_score'], step=1)
                    if st.session_state['player2_score'] or st.session_state['player2_score'] == 0:
                        st.session_state['matchday_input'] = st.date_input("Matchday", st.session_state['matchday_input'] if st.session_state['matchday_input'] else None)
    
        if st.session_state['matchday_input'] and (st.session_state['player1_name'] or st.session_state['player2_name']):
            if st.button("Preview"):
                st.subheader("Confirm the following match result:")
                st.write(f"{st.session_state['player1_name']}: {st.session_state['player1_score']} - {st.session_state['player2_name']}: {st.session_state['player2_score']} on {st.session_state['matchday_input']}")
                st.session_state['show_confirm'] = True
    
        if st.session_state['show_confirm']:
            if st.button("Confirm"):
                spreadsheetId = st.secrets["public_gsheets_id"]
                scope = ['https://www.googleapis.com/auth/spreadsheets']
                google_creds_dict = dict(st.secrets["google_creds"])
                # Create credentials from the dictionary
                credentials = Credentials.from_service_account_info(google_creds_dict, scopes=scope)
                client = gspread.authorize(credentials)
                spreadsheet = client.open_by_key(spreadsheetId)
                worksheet = spreadsheet.sheet1
                match_result = f"{st.session_state['player1_name']} - {st.session_state['player2_name']} {st.session_state['player1_score']}:{st.session_state['player2_score']}"
                game_date = int(st.session_state['matchday_input'].strftime('%Y%m%d'))
                # Append the new game and its date to the worksheet
                worksheet.append_row([match_result, game_date])
                # Clear the inputs and flag data as written
                st.session_state['data_written'] = True               
                # Clear the inputs
                st.session_state['player1_name'] = ''
                st.session_state['player1_score'] = None
                st.session_state['player2_name'] = ''
                st.session_state['player2_score'] = None
                st.session_state['matchday_input'] = None
                st.session_state['show_confirm'] = False
    
                st.experimental_rerun()  # This will rerun the script and update the UI with cleared inputs

display_enter_match_results()
