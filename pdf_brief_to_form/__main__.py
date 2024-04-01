import os
from typing import Union
from dotenv import load_dotenv
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores.chroma import Chroma
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from pydantic.v1.types import SecretStr

from pdf_brief_to_form.file_type import FileType, get_file_type

# init env
load_dotenv()


def response_to_query(vector_index, query: str):
    """
    no doc
    """
    docs = vector_index.similarity_search(query)

    print(f"similarity docs: \n{docs}")

    temperature = os.getenv("OPENAI_TEMPERATURE")
    openai_api_key = SecretStr(os.getenv("OPENAI_API_KEY") or "")

    # temperature=temperature, streaming=True, openai_api_key=openai_api_key

    llm = OpenAI(
        streaming=True, temperature=float(temperature or 0), api_key=openai_api_key
    )

    chain = load_qa_with_sources_chain(llm, chain_type="stuff", verbose=True)

    result = chain.invoke({"input_documents": docs, "question": query})
    print(f"question: {query} \nINVOKE RESULT: \n {result}")


def get_texts_from_md(file_path: str):
    """
    `get_texts_from_md` is provided just for user convenience and should not be overridden.
    """

    loader = UnstructuredMarkdownLoader(file_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(data)

    print(f"Now you have {len(texts)} documents")

    return texts


def get_pages_from_pdf(file_path: str):
    """
    for doc in docs:
        print(str(doc.metadata["page"]) + ":", doc.page_content[:300])
    """
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()

    return pages


def get_vector_index(file_path: str, file_type: Union[FileType, str]):
    embeddings = OpenAIEmbeddings()

    if file_type == FileType.MARKDOWN:
        texts = get_texts_from_md(file_path)
        return Chroma.from_documents(texts, embeddings)

    if file_type == FileType.PDF:
        pages = get_pages_from_pdf(file_path)
        return FAISS.from_documents(pages, embeddings)


def main():
    """
    main function, entrance of the program.
    """

    # We'll split our data into chunks around 500 characters each with a 50 character overlap. These are relatively small.

    queys = [
        "品牌的名字是什么？",
        "报名所需要知道的联系方式信息有什么，比如邮件、电话、地址等？",
        "活动的名称是什么？",
        "给我营销活动描述：提供营销活动的相关信息，例如产品定位、目标受众和核心功能",
        "给出视频的要求：向创作者提供您希望在视频中看到的内容的建议",
    ]

    # change this to your file path
    file_path = "./campaign-info__local__.pdf"

    vector_index = get_vector_index(file_path, get_file_type(file_path))

    for query in queys:
        response_to_query(
            vector_index,
            query,
        )


if __name__ == "__main__":
    main()
