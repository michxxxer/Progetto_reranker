import os
import tempfile

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (PyMuPDFLoader)
from langchain_elasticsearch import ElasticsearchStore
from elasticsearch import Elasticsearch
from langchain_openai import OpenAIEmbeddings
from langchain_voyageai import VoyageAIEmbeddings

def context_extract(documenti: list):
    context = ""
    for indice in documenti:
        context += " " + indice.page_content
    return context

class DocumentLoader:

    def __init__(
            self,
            chunk_size: int = 2300,
            chunk_overlap: int = 250,
            add_start_index: bool = True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.add_start_index = add_start_index

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=self.add_start_index
        )

    def load_and_split_file(self, file_path: str) -> list[Document]:

        tmp = tempfile.NamedTemporaryFile(delete=False)

        with open(file_path, 'rb') as file:
            data = file.read()

        tmp.write(data)
        tmp.close()
        loader = PyMuPDFLoader(tmp.name)
        documents = loader.load()
        return self.text_splitter.split_documents(documents)

class VectorStore:
    def __init__(
            self,
            elasticsearch_url: str,
            elasticsearch_user: str,
            elasticsearch_password: str,
            embedding_model: str,
            index_name: str = "default_index"
    ):

        self.elastic = Elasticsearch(
            hosts=[f"http://{elasticsearch_url}"],
            basic_auth=(elasticsearch_user, elasticsearch_password),
            max_retries=10
        )
        if (embedding_model.startswith("text-embedding")):
            self.embedding = OpenAIEmbeddings(model=embedding_model)
        else:
            self.embedding = VoyageAIEmbeddings(model=embedding_model)

        self.index_name = index_name +"_"+ embedding_model

    def create_index(self, documents: list[Document], index_name: str = None):
        if index_name is None:
            index_name = self.index_name

        self.elastic.indices.create(index=index_name)
        ElasticsearchStore.from_documents(documents, es_connection=self.elastic,
                                          index_name=index_name,
                                          strategy=ElasticsearchStore.ApproxRetrievalStrategy(),
                                          embedding=self.embedding)