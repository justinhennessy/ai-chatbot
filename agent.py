from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain.tools.retriever import create_retriever_tool
from langchain_community.chat_message_histories.upstash_redis import (
    UpstashRedisChatMessageHistory,
)
from langchain.memory import ConversationBufferMemory
from colorama import Fore, Style
import pygame
import time
import requests
import threading
import os
import httpx
from deepgram import (
    DeepgramClient,
    LiveTranscriptionEvents,
    LiveOptions,
)
import os
import getpass
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
import json
import hashlib
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool


# Create Quote retriever
def calculate_checksum(filepath):
    sha256 = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()
    except FileNotFoundError:
        return None

def save_checksum(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f)

def load_checksum(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return {}

def load_file(file, refresh=True):
    current_checksum = calculate_checksum(file)

    if current_checksum is None:
        print(f"File '{file}' not found.")
        return None

    retriever = None
    checksums = load_checksum('file_modification_times.json')

    if file in checksums:
        last_checksum = checksums[file]
        if current_checksum != last_checksum:
            retriever = create_retriever(file)
            checksums[file] = current_checksum
        else:
            retriever = create_retriever(file, False)
    else:
        retriever = create_retriever(file)
        checksums[file] = current_checksum

    save_checksum('file_modification_times.json', checksums)
    return retriever

def create_retriever(file, refresh=True, k=1):
    quote_loader = TextLoader(file)
    docs = quote_loader.load()

    splitter = CharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=0
    )

    splitDocs = splitter.split_documents(docs)

    embedding = OpenAIEmbeddings()

    if refresh:
        print("Refreshing vector database ...")
        vectorStore = FAISS.from_documents(splitDocs, embedding=embedding)
        vectorStore.save_local("faiss_index")
    else:
        print("Loading cached vector database ...")
        vectorStore = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)

    retriever = vectorStore.as_retriever(search_kwargs={"k": k})

    return retriever

# Example usage
retriever = load_file('quotes.txt')

history = UpstashRedisChatMessageHistory(
    url=os.getenv("UPSTASH_URL"),
    token=os.getenv("UPSTASH_TOKEN"),
    session_id="chatbot",
    ttl=0 # expires chats in seconds
)

model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly assistant called Sarah. You have a sassy, flirty personality. DO NOT keep saying you can be asked questions, perhaps only occationally."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    chat_memory=history
)

search = TavilySearchResults()

retriever_tools = create_retriever_tool(
    retriever,
    "myquotes_search",
    "Use this tool when searching for quotes that I have made note of."
)

@tool
def refresh_quote_vectorstore(file='quotes.txt') -> bool:
    """Refreshes the quote vecrtor database after the file has been changed"""
    create_retriever('quotes.txt')
    return True

tools = [search, retriever_tools, refresh_quote_vectorstore]

agent = create_openai_functions_agent(
    llm=model,
    prompt=prompt,
    tools=tools
)

agentExecutor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory
)

def process_chat(agentExecutor, user_input):
    response = agentExecutor.invoke({
        "input": user_input
    })

    return response["output"]

def text_to_speech(text):
    # Define the API endpoint
    url = "https://api.deepgram.com/v1/speak?model=aura-luna-en"

    # Set your Deepgram API key
    api_key = "53ba4e40454bf905b7faf67cb3a6460ca9b2e9aa"

    # Define the headers
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json"
    }

    # Define the payload
    payload = {
        "text": text
    }

    # Make the POST request
    response = requests.post(url, headers=headers, json=payload)

    audio_file_path = "output.mp3"

    # Check if the request was successful
    if response.status_code == 200:
        # Save the response content to a file
        # with open(audio_file_path, "wb") as f:
        #     f.write(response.content)

        with open(audio_file_path, 'wb') as file_stream:
            response = requests.post(url, headers=headers, json=payload, stream=True)
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file_stream.write(chunk)

    else:
        print(f"Error: {response.status_code} - {response.text}")

# Function to run read_response in a separate thread
def play_mp3_in_thread():
    thread = threading.Thread(target=read_response)
    thread.start()

def read_response():
    # Initialize pygame mixer
    pygame.mixer.init()

    # Load the mp3 file
    pygame.mixer.music.load("output.mp3")

    # Play the mp3 file
    pygame.mixer.music.play()

    # Wait for the mp3 file to finish playing
    while pygame.mixer.music.get_busy():
        time.sleep(1)

if __name__ == '__main__':

    while True:
        user_input = input(f"{Fore.GREEN}You:{Fore.WHITE}")
        if user_input.lower() == "exit":
            break

        response = process_chat(agentExecutor, user_input)

        print(f"\n{Fore.BLUE}Sarah:{Fore.WHITE}", response, "\n")
        text_to_speech(response)
        play_mp3_in_thread()