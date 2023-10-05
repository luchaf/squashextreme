from utils import get_name_streaks_df, get_name_opponent_name_df, extract_data_from_games, calculate_combination_stats, derive_results, strip_ansi_escape_codes, capture_stdout, write_prompt_response_to_database
import pandas as pd
from langchain.agents import load_tools, Tool, AgentExecutor
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools.python.tool import PythonAstREPLTool
from langchain.tools.render import render_text_description
from langchain import hub
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
tools = load_tools(["llm-math", "wikipedia", "terminal"], llm=llm)
python = PythonAstREPLTool(locals={"df_tmp": df_tmp,
                                   "df_players_stats": df_players_stats,
                                   "df_combination_stats": df_combination_stats,
                                   "df_streaks": df_streaks,
                                  })
python_pandas_tool = Tool(
            name="pythonastrepltool",
            func=python.run,
    description = f"""     
        # If you want to create your own analysis, you could just access the df dataframe directly.
        # If you want to use existing statistics, like win or loss streaks you could use the streaks dataframe.
        
        Here is a description of the df_tmp dataframe:
        The df_tmp dataframe has the following columns: {columns_df_tmp}.
        Each row of the df_tmp dataset represents the outcome of a squash game.
        The 'First Name' column contains the name of the first player.
        The 'Second Name' column contains the name of the second player.
        The "First Score" column contains the score of the first player.
        The "Second Score" column contains the score of the second player.
        The "date" column is the date on which the game was played.
                
        Here is a description of the 'df_players_stats' dataframe: 
        The 'df_players_stats' columns are: {columns_df_players_stats}.
        The 'df_players_stats' columns represent the number of wins and the total score for each player.
        The 'df_players_stats' dataframe consists of one row per player. 
        The index of the dataframe represents the names of the players: Simon, Lucas, and Friedemann. 
        The 'Wins' column of 'df_players_stats' dataframe contains the number of wins for each player.
        The 'Total Score' column of 'df_players_stats' dataframe contains the total score for each player. 
        
        Here is a description of the 'df_combination_stats' dataframe: 
        The 'df_combination_stats' columns are: {columns_df_combination_stats}
        The dataframe 'df_combination_stats' contains statistics about different player combinations in a game. 
        Each row represents a unique pair of players, indicated by the 'Player Combo' column. 
        The 'Total Score A' and 'Total Score B' columns represent the total 
        scores of the first and second players in the pair, respectively. 
        The 'Wins A' and 'Wins B' columns represent the number of wins for the first and second players, respectively. 
        The 'Balance' column represents the difference in wins between the two players, 
        with a negative number indicating that the second player has more wins.
        
        Here is a description of the 'df_streaks' dataframe: 
        The 'df_streaks' columns are: columns_streaks
        "The 'df_streaks' dataframe contains information about the longest win and loss streaks.
        Each row represents a different type of streak: the longest win streak and the longest loss streak. 
        The columns represent the individuals and contain the length of their respective streaks. 
        
        All dates in the dataframes are stored like this 20210821 as integers so in the yyyymmdd integers form. 
        
        Run python pandas operations on these dataframes to help you get the right answer.
        """
        )
tools.append(python_pandas_tool)
prompt = hub.pull("hwchase17/react-chat")
prompt = prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)
llm_with_stop = llm.bind(stop=["\nObservation"])
agent = {
    "input": lambda x: x["input"],
    "agent_scratchpad": lambda x: format_log_to_str(x['intermediate_steps']),
    "chat_history": lambda x: x["chat_history"]
} | prompt | llm_with_stop | ReActSingleInputOutputParser()
memory = ConversationBufferMemory(memory_key="chat_history")

st.title("Chat with the SquashBot")
st.write(random.choice(st.secrets["squash_bot_intros"]))

if "messages" not in st.session_state:
    st.session_state.messages = []

# Check if memory is already in session_state, if not initialize it
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history")

# Use the memory state in AgentExecutor
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

