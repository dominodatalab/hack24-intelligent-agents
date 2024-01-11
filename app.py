import json
import mlflow
import os
import pandas as pd
import pickle
import requests
import subprocess
import streamlit as st
import time
import warnings

from datetime import datetime
from datetime import timedelta

from domino import Domino
from IPython.core.magic import register_line_magic, needs_local_scope
from pydantic import BaseModel, Field

from langchain import hub
from langchain.chains import LLMMathChain
from langchain import memory as lc_memory
from langchain.agents import AgentExecutor, AgentType, initialize_agent, create_csv_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, StructuredTool, Tool, tool, DuckDuckGoSearchRun
from langchain.schema import HumanMessage, SystemMessage


from rag import get_condense_prompt_qa_chain


warnings.filterwarnings('ignore')
llm = ChatOpenAI(temperature=0)

search = DuckDuckGoSearchRun()

# Load the embeddings from the pickle file; change the location if needed
if 'store' not in locals() or store is None:
    with open("faiss_doc_store.pkl", "rb") as f:
        store = pickle.load(f)

qa = get_condense_prompt_qa_chain(store)

search_tool = Tool.from_function(
    func=search.run,
    name="Search",
    description="useful for when you need to search the internet for information"
)

llm_math_chain = LLMMathChain.from_llm(ChatOpenAI(temperature=0))

math_tool = Tool.from_function(
    func=llm_math_chain.run,
    name="Calculator",
    description="Useful for when you are asked to perform math calculations"
)

# RAG tool to answer queries about Domino
@tool("ddl_rag", return_direct=True)
def ddl_rag(user_query:str) -> str:
    """Answer questions about Domino product features, technical or ambigous questions"""
    if not user_query:
        return None
    return qa({"question": user_query})['answer']

# create an experiment in MLflow
@tool("create_experiment", return_direct=True)
def create_experiment(experiment_name:str):
    """Create an experiment in Domino"""
    # we'll make the name unique in the project by appending a timestamp so that you and other users can run this cell more than once.
    timestamp = time.time()
    if experiment_name:
        experiment_name = f"{experiment_name}-{timestamp}"
        # below, we'll use the returned experiment_id in calls to mlflow.start_run() to add data to the experiment.
        experiment_id = mlflow.create_experiment(experiment_name.replace("experiment_name = \"", ""))
        print(f"Experiment id: {experiment_id}")
        print(f"Experiment name: {experiment_name}")


# execute a blocking job in Domino
@tool("create_run_job", return_direct=True)
def create_run_job(job_file:str):
    """Create and run  a job in Domino"""

    owner=os.getenv("DOMINO_PROJECT_OWNER")
    project=os.getenv("DOMINO_PROJECT_NAME")

    domino = Domino(
    f"{owner}/{project}",
    api_key=os.environ["DOMINO_USER_API_KEY"],
    host=os.environ["DOMINO_API_HOST"],
    )
    domino.authenticate(domino_token_file = os.getenv('DOMINO_TOKEN_FILE'))
    if job_file:
        # Blocking: this will start the run and wait for the run to finish before returning the status of the run
        domino_run = domino.runs_start_blocking(
            [job_file], title="Started from Domino agent"
        )
    print(domino_run)

# analyze control centre costs
@tool("analyze_costs", return_direct=True)
def analyze_costs(user_query:str) -> str:
    """Answer usage and cost analytics related questions for this deployment"""
    if not user_query:
        return None
    # get all the data and write it to a csv
    csv_file_name = 'control_centre_costs.csv'
    API_KEY = os.environ["DOMINO_USER_API_KEY"]
    URL = "https://johnal33586.marketing-sandbox.domino.tech/v4/gateway/runs/getByBatchId"
    headers = {'X-Domino-Api-Key': API_KEY}
    last_date = datetime.now().strftime('%Y-%m-%d')
    
    last_date = datetime.strftime(
        datetime.strptime(last_date, '%Y-%m-%d') + timedelta(days = 1),
        '%Y-%m-%d',
    )
    
    try:
      os.remove(csv_file_name)
    except:
      pass
    
    batch_ID_param = ""
    while True:
        batch = requests.get(url = URL + batch_ID_param, headers = headers)
        parsed = json.loads(batch.text)
        batch_ID_param = "?batchId=" + parsed['nextBatchId']
        df = pd.DataFrame(parsed['runs'])
        df[df.endTime <= last_date].to_csv(
            csv_file_name,
            mode="a+",
            index=False,
            header=True)
        if len(df.index) < 1000 or len(df.index) > len(df[df.endTime <= last_date].index):
            break
            
    # create a csv agent
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    agent_csv = create_csv_agent(
    llm,
    "control_centre_costs.csv",
    agent_type=AgentType.OPENAI_FUNCTIONS,
    verbose=False
    )
    return agent_csv.run(user_query)

# analyze organization costs
@tool("analyze_org_costs", return_direct=True)
def analyze_org_costs(user_query:str) -> str: 
    """Answer usage and cost questions related to organizations for admins"""
    # Set your API key and domain
    API_KEY = os.environ["DOMINO_USER_API_KEY"]
    DOMAIN = 'johnal33586.marketing-sandbox.domino.tech'
    COST_SESSION_COOKIE = '76bc06ba-6b89-40c0-bb21-070b9eedc52d'
    
    # URL for the request
    url = f'https://{DOMAIN}/v4/controlCenter/utilization/statsForAllOrganizations'
    
    # Headers and parameters for the request
    headers = {
        'X-Domino-Api-Key': API_KEY,
        'Accept': 'application/json',
        'Cookie': f'dominoSession={COST_SESSION_COOKIE}'
    }
    #print(headers)
    params = {
        'startDate': '20240101',
        'endDate': '20240109'
        # 'aggregate': 'label:johnal33586.marketing-sandbox.domino.tech.com/659c6bded542310fd7b03f55'
    }
    
    # Making the GET request
    response = requests.get(url, headers=headers, params=params)
    # Checking if the request was successful
    if response.status_code == 200:
        csv_file_name = "organization_costs.csv"
        # Print response JSON or process it as needed
        cost_stats = response.json()
        # print(cost_stats)
        df = pd.json_normalize(cost_stats)
        df.to_csv(csv_file_name, index=False)
        
    # create a csv agent
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    agent_csv = create_csv_agent(
    llm,
    "organization_costs.csv",
    agent_type=AgentType.OPENAI_FUNCTIONS,
    verbose=False
    )
    return agent_csv.run(user_query)


# execute a blocking job in Domino
@tool("generate_code", return_direct=True)
def generate_code(task:str):
    '''Generates code and saves it to a file'''
    if not task:
        return None
    messages = [
    SystemMessage(
        content="You are a helpful assistant that generates code.just output code without any commentary"
        ),
    HumanMessage(
        content= task
        ),
    ]
    # Writing to the file
    timestamp = time.time()
    code_file_name = f"agent-code-{timestamp}.py"
    subprocess.run(["touch", code_file_name])
    with open(code_file_name, 'w') as file:
        file.write(llm(messages).content)
    

tools = [search_tool, math_tool, create_experiment, create_run_job, analyze_costs, ddl_rag, generate_code]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False
)

st.set_page_config(
    page_title="Orbit assistant",
)

st.subheader("🛠️ Your helpful AI assistant")

memory = lc_memory.ConversationBufferMemory(
    chat_memory=lc_memory.StreamlitChatMessageHistory(key="langchain_messages"),
    return_messages=True,
    memory_key="chat_history",
)

if st.sidebar.button("Clear message history"):
    print("Clearing message history")
    memory.clear()

for msg in st.session_state.langchain_messages:
    avatar = "💫" if msg.type == "ai" else None
    with st.chat_message(msg.type, avatar=avatar):
        st.markdown(msg.content)

if prompt := st.chat_input(placeholder="Ask me a question!"):
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant", avatar="💫"):
        message_placeholder = st.empty()
        with st.spinner("Running..."):
            response = agent.run(prompt)
        # Define the basic input structure for the chains
        input_dict = {"input": prompt}
        memory.save_context(input_dict, {"output": response})
        if response:
            message_placeholder.markdown(response)


