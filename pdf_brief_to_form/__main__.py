from typing import List
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores.chroma import Chroma
from langchain_community.vectorstores.pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

from langchain_openai import ChatOpenAI

import os

# init env
load_dotenv()


def pineconeForCloud():
    raise ("to complete PineconeForCloud: \n")


def localChroma(texts: List[Document], embeddings: OpenAIEmbeddings, query: str):
    print("local it into Chroma: \n")
    vectorStore = Chroma.from_documents(texts, embeddings)

    docs = vectorStore.similarity_search(query)

    # Here's an example of the first document that was returned
    for doc in docs:
        print(f"{doc.page_content}\n")

    OPENAI_TEMPERATURE = os.getenv("OPENAI_TEMPERATURE")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    llm = ChatOpenAI(temperature=OPENAI_TEMPERATURE, openai_api_key=OPENAI_API_KEY)

    chain = load_qa_with_sources_chain(llm, chain_type="stuff")

    docs = vectorStore.similarity_search(query)

    result = chain.invoke(
        {"input_documents": docs, "question": query}, return_only_outputs=True
    )
    print(f"INVOKE RESULT: \n {result}")


def main():
    print("main to go!")
    loader = UnstructuredMarkdownLoader(file_path="./campaign-info__local__.md")

    data = loader.load()

    # Note: If you're using PyPDFLoader then it will split by page for you already
    print(f"You have {len(data)} document(s) in your data")
    print(f"There are {len(data[0].page_content)} characters in your sample document")
    print(f"Here is a sample: {data[0].page_content[:200]}")

    # We'll split our data into chunks around 500 characters each with a 50 character overlap. These are relatively small.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(data)

    print(f"Now you have {len(texts)} documents")

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    query = "活动营销的核心诉求是什么?"

    localChroma(texts, embeddings, query)


if __name__ == "__main__":
    main()
