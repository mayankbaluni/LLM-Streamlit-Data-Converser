import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from langchain.agents import AgentType
from langchain.tools.python.tool import PythonAstREPLTool
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI, ChatGooglePalm
from langchain.llms import GooglePalm, OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_pandas_dataframe_agent
import streamlit as st
from streamlit_chat import message
import os
import statsmodels as sm
import seaborn as sns
import os
import sys
from io import StringIO, BytesIO




#api_key1 = st.secrets["GOOGLE_API_KEY"]
#api_key2 = st.secrets["OPENAI_API_KEY"]
os.environ["GOOGLE_API_KEY"] ="AIzaSyD29fEos3V6S2L-AGSQgNu03GqZEIgJads"
#os.environ["OPENAI_API_KEY"] = api_key2
llm = GooglePalm(temperature=0.9, max_output_tokens= 512,verbose=True,streaming=True)
#llm = OpenAI(temperature=0.9,verbose=True)



if 'prompts' not in st.session_state:
    st.session_state.prompts = []
if 'responses' not in st.session_state:
    st.session_state.responses = []


def generate_code(prompt, data_type, missing, shape):
    

    prompt_template = PromptTemplate(
    input_variables=['prompt','data_type', 'shape', 'missing'],
        template="Company Data is loaded as 'df', column names and their types: {data_type}\n\
        df.shape= {shape}\
        missing values: {missing}\
        instructions: Please provide short code in 2 lines for data analysis, user knows python, include column names and types,. Answer queries like the example below:\
        \
        Example 1:\
        query: which device has mostly asset status as stationary?.\
        Answer:\
        search 'Device Name' with 'Asset Status' as 'Stationary'. use 'Device Name' and 'Asset Status' and their types are 'str', 'str' and missing values are '357' and '332'\
        Count the unique values of 'Asset Status' to determine how many distinct statuses exist, .\
        \
        Example 2:\
        query: what are the number of sales of 2curex in 2019?.\
        Answer:\
        Get 2019 sales for '2cureX', use str.lower().str.contains('2curex') & (df['Year'] == 2019) \
        find net profit of company '2cureX', you need 'net profit', 'Year' and company 'Name' columns, their types are 'int', 'int', 'str'\
        \
        Example 3:\
        query: mean of state of charge.\
        Answer:\
        calculate mean upto two decimal place, use df['State Of Charge%'].mean().round(2) \
        Display answer\
        \
        Example 4:\
        query: which asset has the maximum state of charge?.\
        Answer:\
        find name of the 'Asset Name' which has maximum state of charges use code(df['State Of Charge%'].idxmin())) \
        give out the asset name and state of charge'\
        \
        Example 5:\
        query: what is the median value of state of charge when TRU status is off.\
        Answer:\
        filter by TRU Status = off, find median of 'State Of Charge%' use code(df[(df['TRU Status'] == 'On')]['State Of Charge%'].median()) \
        give out the median value'\
        \
        Example 6:\
        query: graphs between average state of charge vs charging status.\
        Answer:\
        make a plot between 'Carging Status' and average 'State Of Charge%'. use matplotlib, use code (df.groupby('Charging Status')['State Of Charge%'].mean().plot(kind='bar')) \
        present the plot'\
        if query is not in example, make similar example and include code like example2, example2, example4\
        \
        query: {prompt}\
        Answer: \
        " 
    )
    about_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="about")


    chain = SequentialChain(chains=[about_chain], input_variables=["prompt","data_type", "shape", "missing"], output_variables=["about"])

    response = chain.run({'prompt': prompt, 'data_type': data_type, 'shape': shape, 'missing':missing})
    return response
    




st.set_page_config(page_title="Data Analyst", page_icon="chart_with_upwards_trend")
st.title(':blue[Data Analysis ] :red[Chatbot]')

file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
}


def clear_submit():
    """
    Clear the Submit Button State
    Returns:

    """
    st.session_state["submit"] = False


@st.cache_data()
def load_data(uploaded_file):
    try:
        ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    except:
        ext = uploaded_file.split(".")[-1]
    if ext in file_formats:
        return file_formats[ext](uploaded_file)
    else:
        st.error(f"Unsupported file format: {ext}")
        return None


with st.sidebar:
    uploaded_file = st.file_uploader(
    "Upload a Data file",
    type=list(file_formats.keys()),
    help="Various File formats are Support",
    on_change=clear_submit,
)

if not uploaded_file:
    st.warning(
        "Please upload your file"
    )


if uploaded_file:
    df = load_data(uploaded_file)

    with st.sidebar:
        st.subheader("Your Data")
        st.dataframe(df)


    if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt1 := st.chat_input(placeholder="How many rows in this data?"):

        st.session_state.messages.append({"role": "user", "content": prompt1})
        st.chat_message("user").write(prompt1)

        data = df.head()
        missing = df.isnull().sum()
        shape = df.shape
        columns= df.columns

        variable_type_info = []

        for key, value in data.items():
            variable_type = type(value)
            data_type1 = f"'{key}' is of type: {variable_type}"
            variable_type_info.append(data_type1)
        data_type = "\n".join(variable_type_info)

        prompt =  generate_code(prompt1, missing, shape, columns) 
        print(prompt)


        memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", human_prefix= "", ai_prefix= "")

        for i in range(0, len(st.session_state.messages), 2):
            if i + 1 < len(st.session_state.messages):
                current_message = st.session_state.messages[i]
                next_message = st.session_state.messages[i + 1]
                
                current_role = current_message["role"]
                current_content = current_message["content"]
                
                next_role = next_message["role"]
                next_content = next_message["content"]
                
                # Concatenate role and content for context and output
                context = f"{current_role}: {current_content}\n{next_role}-said: {next_content}"
                
                memory.save_context({"question": context}, {"output": ""})
        
        #llm1= ChatOpenAI(temperature=0.7,  model="gpt-3.5-turbo-0613", streaming=True, verbose = True) #incase we need openai
        
        #agent = create_pandas_dataframe_agent(llm1 ,df, agent_type=AgentType.OPENAI_FUNCTIONS
        agent = create_pandas_dataframe_agent(llm ,df
                                              ,prefix="You are an expert data analyst. You need to perform analysis on company's data 'df' as told by supervisor, make necessary changes.\
                                               Answer nicely for a non technical person in simple sentence or in table, present it professionally.'.\
                                                if you cant find the result in 10 tries, just say 'Use Correct Column Names'"
        ,handle_parsing_errors=True,verbose=True, number_of_head_rows = 10
            )

        message = st.chat_message("assistant")
        with message:
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
            try:
                # Your code that may raise an error here
                old_stdout = sys.stdout
                sys.stdout = captured_output = StringIO()
                
                response = agent.run(prompt, callbacks=[st_cb])
                fig = plt.gcf()
                if fig.get_axes():
                            # Adjust the figure size
                    fig.set_size_inches(12, 6)

                    # Adjust the layout tightness
                    plt.tight_layout()
                    buf = BytesIO()
                    fig.savefig(buf, format="png")
                    buf.seek(0)
                    
                    #message.write("Hello human")
                    st.image(buf, caption={prompt1},use_column_width=True)
                    
                                        
                    st.session_state.messages.append({"role": "assistant", "content": f"broken-{prompt1}"})
                    st.stop()  

                sys.stdout = old_stdout
            
            except Exception as e:
                # Handle the error here
                st.error("Problem in the Data! Please Try Again with a different question.")
                st.stop()  # Stop execution to prevent further code execution
            st.session_state.messages.append({"role": "assistant", "content": response})     
            st.markdown(response)

       

            
       



