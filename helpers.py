import os
import json
import hashlib
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.tools import BaseTool, StructuredTool, tool

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
