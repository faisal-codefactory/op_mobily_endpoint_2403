
# Implemented on March 15, 2024 
# run the serrver and use http://localhost:9999/docs to  test endpoints
# Importing required libraries
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import File, UploadFile, status
import random, string
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
# For evaluation of RAG
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
import uvicorn

load_dotenv()

app = FastAPI()

openai_api_key = os.getenv("OPENAI_API_KEY")
assert openai_api_key is not None, "OpenAI API key is not set"

pinecone_api_key = os.getenv("PINECONE_API_KEY")
assert pinecone_api_key is not None, "Pinecone API key is not set"

embed_model_name = os.getenv("EMBED_MODEL")


origins = ["*"]
app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins, 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
    )

def init_pinecone_index(index_name:str):
    pine_client = Pinecone(api_key=pinecone_api_key)
    serverless_spec = ServerlessSpec(cloud="aws", region = "us-west-2")
    for ind in pine_client.list_indexes():
        if ind['name'] == index_name:
            pine_client.delete_index(index_name)
            print("deleted previous index")
    try:         
        pine_client.create_index(name=index_name,
                                      dimension=384,
                                      spec=serverless_spec,
                                      metric='cosine')
    except Exception as e:
        raise Exception(f"Unable to create index. Error: {str(e)}")

def delete_previous_index(index_name:str):
    try:    
        pine_client = Pinecone(api_key=pinecone_api_key)
        for ind in pine_client.list_indexes():
            if ind['name'] == index_name:
                pine_client.delete_index(index_name)
                print("deleted previous index")
            #else:
            #    print(f"index name {index_name} not found. Safe to continue")    
    except Exception as e:
        raise Exception(f"Unable to delete index. Error: {str(e)}")            


embed_fn = HuggingFaceEmbeddings(model_name = embed_model_name)

def create_qa_chain_from_pinecone_index(index_name:str):
    try:    
        pine_client = Pinecone(api_key=pinecone_api_key)
        index_names = [ind['name'] for ind in pine_client.list_indexes()]
        assert index_name in index_names, f"index name {index_name} not found."
    except Exception as e:
        raise Exception(f"Unable to find pinecone index: {index_name}. Error: {str(e)}")
    
    try:    
        vecdb = PineconeVectorStore.from_existing_index(index_name, embed_fn)
        prompt_str_maintain = """
            You are an AI Assistant. Your job is to assist in question-answering tasks.
            Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know the answer politely. 

            Question: {question} 
            Context: {context} 
            Answer:
        """
        prompt_str_mobily = """  
            You are an AI Assistant. Your job is to answer the question like you are Sales Agent talking to a potential investor/customer.
            Use the following pieces of retrieved context to answer the question. 
            If the question is regarding potential investment, do the following:
                1- Include the relevant Historical Financial Performance of the company in the answer
                2- Include relevant Facts and Figures to support the answer
                3- Include the future plans of the company in the answer if necessary
                4- Add the contact details ( Ph: 00966560314099 email: IRD@mobily.com.sa ) of the Investors Relations Department.
            
            If you don't know the answer, just say that you don't know the answer politely and Add the contact details ( Ph: 00966560314099 email: IRD@mobily.com.sa ) of the Investors Relations Department.
            Question: {question} 
            Context: {context} 
            Answer:    
            """

        if 'mobily' in index_name:
            prompt_str = prompt_str_mobily
        else :
            prompt_str = prompt_str_maintain

        chat_prompt = ChatPromptTemplate.from_template(prompt_str)
        num_chunks = 5
        chat_llm = ChatOpenAI(model = "gpt-3.5-turbo", openai_api_key = openai_api_key, temperature=0.1)
        retriever = vecdb.as_retriever(search_type="similarity", search_kwargs={"k": num_chunks})
        retrieval = RunnableParallel({"question": RunnablePassthrough(), "context": retriever})
        qa_chain = (retrieval | chat_prompt | chat_llm)
        return qa_chain
    except Exception as e:
        raise Exception(f"Unable to create qa chain. Error: {str(e)}")

index_name_opman = "op-maintain-manual"
qa_chain_opman= create_qa_chain_from_pinecone_index(index_name_opman)

index_name_mobily = "mobily-ar-2022"
qa_chain_mobily= create_qa_chain_from_pinecone_index(index_name_mobily)

@app.get("/", response_class=HTMLResponse)
def index():
    message = "Welcome to RAG Assistive AI"
    html_content = f"<html><body><h1>{message}</h1></body></html>"
    return HTMLResponse(content=html_content)



@app.post("/qa_from_op_maintenance_pdf")
async def qa_from_op_maintenance(user_query: str ):
    if user_query is None:
        return JSONResponse({"message": "Query is empty. Please enter valid query."})
    try:
        out = await qa_chain_opman.ainvoke(user_query)
        #print(out)
        return JSONResponse(content={"message": "Response Generated Successfully!", "Response": out.content}, status_code=status.HTTP_200_OK)    
    except Exception as ex:
        return JSONResponse(content={"error": str(ex)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.post("/qa_from_mobily_pdf")
async def qa_from_mobily(user_query: str ):
    if user_query is None:
        return JSONResponse({"message": "Query is empty. Please enter valid query."})
    try:
        out = await qa_chain_mobily.ainvoke(user_query)
        print(out)
        return JSONResponse(content={"message": "Response Generated Successfully!", "Response": out.content}, status_code=status.HTTP_200_OK)    
    except Exception as ex:
        return JSONResponse(content={"error": str(ex)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


if __name__ == "__main__":
    # This line enables listening on all network interfaces.
    # This change is introduces to access running swagger of WSL in windows browser.
    uvicorn.run(app, host="0.0.0.0", port=9999)