from utils import get_name_streaks_df, get_name_opponent_name_df, extract_data_from_games, calculate_combination_stats, derive_results, strip_ansi_escape_codes, capture_stdout, write_prompt_response_to_database
import pandas as pd
from langchain.agents import load_tools, Tool, AgentExecutor
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
import streamlit as st
import random

def get_response():
    return agent_executor.invoke({"input": prompt})['output']

# Existing code for statistics:
df_sheet = pd.read_csv(st.secrets["public_gsheets_url"])   
df_sheet['parsed_sheet_df'] = df_sheet.apply(lambda x: extract_data_from_games(x["games"], x["date"]), axis=1)
df_tmp = pd.DataFrame()
for _, row in df_sheet.iterrows():
    df_tmp = pd.concat([df_tmp, row["parsed_sheet_df"]])
df = df_tmp.copy()
df = df.reset_index(drop=True).copy()
df = df.reset_index()
# Derive player and combination stats
df_combination_stats = calculate_combination_stats(df)
df = get_name_opponent_name_df(df)

# Calculate individual stats
df_players_stats = df.groupby('Name').agg({'Wins': 'sum', 'Player Score': 'sum'}).rename(
    columns={'Player Score': 'Total Score'}).sort_values("Wins", ascending=False)

# Derive results
results = derive_results(df)

# Calculate win and loss streaks
# streaks = calculate_streaks(results)
df_streaks = get_name_streaks_df(df)
df_streaks = df_streaks.sort_values(["longest_win_streak", "longest_loss_streak"], ascending=False)

columns_df_tmp = df_tmp.columns
columns_df_players_stats = df_players_stats.columns
columns_df_combination_stats = df_combination_stats.columns
columns_df_streaks = df_streaks.columns

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0, openai_api_key=st.secrets["open_ai_key"])
tools = load_tools(["llm-math", "wikipedia"], llm=llm)

llm_with_stop = llm.bind(stop=["\nObservation"])

memory = ConversationBufferMemory(memory_key="chat_history")

st.title("Chat with the SquashBot")
st.write(random.choice(st.secrets["squash_bot_intros"]))

if "messages" not in st.session_state:
    st.session_state.messages = []

# Check if memory is already in session_state, if not initialize it
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history")

# Use the memory state in AgentExecutorprompt = prompt.partial(
llm_with_stop = llm.bind(stop=["\nObservation"])
prompt="ok"
agent = {
    "input": lambda x: x["input"],
    "agent_scratchpad": lambda x: format_log_to_str(x['intermediate_steps']),
    "chat_history": lambda x: x["chat_history"]
} | prompt | llm_with_stop | ReActSingleInputOutputParser()
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=st.session_state.memory)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
if prompt := st.chat_input("Ask and thou ball receive"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):      
        # Use the capture_stdout function to get both the response and printed content
        response, printed_output = capture_stdout(get_response)      
        # Then, write the response to Streamlit
        st.write(response)
        # First, write any printed content to Streamlit
        # Strip ANSI escape codes from printed_output
        printed_output_cleaned = strip_ansi_escape_codes(printed_output)
        if printed_output_cleaned:
            with st.expander("Dive into my brainwaves to understand why I served up this answer! 🏸🌊🧠"):
                st.text(printed_output_cleaned)
        # write prompt and response to database
        write_prompt_response_to_database(prompt, response)
    st.session_state.messages.append({"role": "assistant", "content": response})

