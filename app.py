import datetime
import streamlit as st

title = st.text_input('copy paste game results from whatsapp chat here:')
#st.write('Game results', title)

d = st.date_input("Day to document the results:", datetime.datetime.now())
st.write('Day to document:', d)
