import requests
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from helpers import load_file
from langchain.tools import BaseTool, StructuredTool, tool

search = TavilySearchResults()
retriever = load_file('quotes.txt')

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

def append_quote_to_file(quote_file, new_quote):
    with open(quote_file, 'a') as file:
        file.write(']\n' + new_quote + ']\n')

@tool
def add_new_quote():
    """This allows users to add new quotes to the quotes.txt file"""
    quote_file = 'quotes.txt'
    new_quote = input("Enter the new quote: ")
    append_quote_to_file(quote_file, new_quote)
    print("New quote added successfully.")

tools = [search, retriever_tools, refresh_quote_vectorstore, add_new_quote]

@tool
def get_mercury_retrograde_status():
    """This tell you if we are in a mercury retrograde or not"""
    url = "https://mercuryretrogradeapi.com/"
    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        print("Failed to fetch Mercury retrograde status.")
        return None
