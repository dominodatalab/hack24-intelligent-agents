import json
import mlflow
import os
import pandas as pd
import pickle
import requests
import subprocess
import time
import warnings

from datetime import datetime
from datetime import timedelta

from domino import Domino
from IPython.core.magic import register_line_magic, needs_local_scope
from pydantic import BaseModel, Field

from langchain import hub
from langchain.chains import LLMMathChain
from langchain.agents import AgentExecutor, AgentType, initialize_agent, create_csv_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, StructuredTool, Tool, tool, DuckDuckGoSearchRun

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


tools = [search_tool, math_tool, create_experiment, create_run_job, analyze_costs, ddl_rag]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False
)


@register_line_magic
@needs_local_scope
def orbit(user_query, local_ns=None):
    """A line magic that calls the Orbit agent."""
    if not user_query:
        return None
    return agent.run(user_query)



