from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent, AgentExecutor
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
from tools import search, retriever_tools, refresh_quote_vectorstore, add_new_quote, get_mercury_retrograde_status

# Example usage
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

tools = [search, refresh_quote_vectorstore, retriever_tools, add_new_quote]

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    chat_memory=history
)

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
        # Save the response content to a file in chucks
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