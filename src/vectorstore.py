from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from typing import List


class VectorStore:
    """
    A class for managing a vector store using FAISS, with document loading, splitting, and embedding functionalities.

    Attributes:
    ----------
    __embedding : OllamaEmbeddings
        Embedding model for converting text into vector representations. Defaults to 'nomic-embed-text' if not provided.
    __chunk_size : int
        Size of each document chunk used during splitting for embedding.
    __chunk_overlap : int
        Overlap between document chunks for effective information retention.
    __vectorestore : FAISS
        FAISS vector store instance for managing document vectors.
    __retriever : Retriever
        Retriever instance for document search based on vector similarity.

    Methods:
    -------
    load_docs(files_path: List[str]) -> List[Document]
        Loads PDF documents from provided file paths.
    
    split_docs(docs: List[Document]) -> List[Document]
        Splits documents into chunks based on chunk size and overlap.

    create_vectorsore(files_path: List[str]) -> FAISS
        Creates the FAISS vector store from document embeddings.

    clear_vectorstore() -> None
        Clears all documents from the vector store.

    add_to_vectorstore(files_path: List[str]) -> None
        Adds new documents to the existing vector store.

    delete_docs() -> None
        Placeholder method for document deletion from the vector store.
    """

    def __init__(self, embedding: OllamaEmbeddings = None, chunk_size: int = 250, chunk_overlap:int =10):
        self.__embedding = embedding if embedding is not None else OllamaEmbeddings(model = 'nomic-embed-text',)
        self.__chunk_size = chunk_size
        self.__chunk_overlap = chunk_overlap
        self.__vectorestore = None
        self.__retriever = None
    
    
    def load_docs(self, files_path: List[str]) -> List[Document]:
        """
        Loads PDF documents from the specified file paths.

        Parameters:
        ----------
        files_path : List[str]
            A list of file paths for the PDF documents to load.

        Returns:
        -------
        List[Document]
            A list of loaded Document objects.
        """

        docs = [PyPDFLoader(file_path = file_path,extract_images = False).load() for file_path in files_path]
        return docs
    
    def split_docs(self, docs: List[Document]) -> List[Document]:
        """
        Splits documents into chunks based on chunk size and overlap.

        Parameters:
        ----------
        docs : List[Document]
            A list of Document objects to split.

        Returns:
        -------
        List[Document]
            A list of split Document objects.
        """
        docs_list = [item for sublist in docs for item in sublist]
        # docs_list = [item for item in docs]
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=self.__chunk_size, chunk_overlap=self.__chunk_overlap)
        doc_splits = text_splitter.split_documents(docs_list)
        
        return doc_splits
    


    def create_vectorsore(self, files_path: List[str]) -> FAISS:
        """
        Creates the FAISS vector store by loading, splitting, and embedding documents.

        Parameters:
        ----------
        files_path : List[str]
            A list of file paths for the documents to add to the vector store.

        Returns:
        -------
        FAISS
            The created FAISS vector store instance.
        """
        docs = self.load_docs(files_path = files_path)
        doc_splits = self.split_docs(docs = docs)
        self.__vectorestore  = FAISS.from_documents(documents=doc_splits,embedding=self.__embedding)
        self. __retriever = self.__vectorestore.as_retriever()

    def clear_vectorstore(self) -> None:
        """
        Clears all documents from the vector store, deleting stored vectors and resetting the retriever.
        """
        if self.__vectorestore is not None and self.__vectorestore.index.ntotal !=0:
            self.__vectorestore.delete([self.__vectorestore.index_to_docstore_id[i] for i in range(self.__vectorestore.index.ntotal)])
        self.__retriever = None
 
    def add_to_vectorstore(self, files_path:  List[str]):
        """
        Adds new documents to the existing vector store by loading and splitting them, then embedding their chunks.

        Parameters:
        ----------
        files_path : List[str]
            A list of file paths for the new documents to add.
        """
        docs = self.load_docs(files_path = files_path)
        doc_splits = self.split_docs(docs = docs)
        self.__vectorestore.add_documents(documents = doc_splits)


    def delete_docs(self):
        pass

    # --------------------------- Properties ----------------------------------
    @property
    def embedding(self) -> OllamaEmbeddings:
        return self.__embedding
    
    @embedding.setter
    def embedding(self, embedding: OllamaEmbeddings)-> None:
        self.__embedding = embedding

    @property
    def chunk_size(self) -> int:
        return self.__chunk_size
    
    @chunk_size.setter
    def chunk_size(self, chunk_size_:int) -> None:
        if not isinstance(chunk_size_, int):
            raise ValueError("chunk_size must be an integer")
        if chunk_size_ < 1:
            raise ValueError("chunk_size must be greater than 1")
        self.__chunk_size =chunk_size_

    @property
    def chunk_overlap(self) -> int :
        return self.__chunk_overlap
    
    @chunk_overlap.setter
    def chunk_overlap(self, chunk_overlap_: int) -> None:
        if not isinstance(chunk_overlap_, int):
            raise ValueError("chunk_overlap_ must be an integer")
        if chunk_overlap_ < 0:
            raise ValueError("chunk_overlap_ must be none negative")
        self.__chunk_overlap = chunk_overlap_

    @property
    def retriever(self):
        return self.__vectorestore.as_retriever() if self.__vectorestore is not None else None
    
    @property
    def vectorstore(self):
        return self.__vectorestore

