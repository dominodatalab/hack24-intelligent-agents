import pickle

from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores.base import VectorStoreRetriever


_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """You are an AI assistant for answering questions about information in Domino Data Labs product documentation.
You are given the following extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not about AI or ML or data science or MLOps or related to Domino Data Lab, politely inform them that you are tuned to only answer questions about MLOps, data science and Domino Data Lab.
Question: {question}
=========
{context}
=========
Answer in Markdown:"""
QA_PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])

def get_condense_prompt_qa_chain(store):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)
    # see: https://github.com/langchain-ai/langchain/issues/5890
    model = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=VectorStoreRetriever(vectorstore=store),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT})
    return model

# Load the embeddings from the pickle file; change the location if needed
if 'store' not in locals() or store is None:
    with open("faiss_doc_store.pkl", "rb") as f:
        store = pickle.load(f)

# qa = get_condense_prompt_qa_chain(store)
# result = qa({"question": "What is a Domino data set"})