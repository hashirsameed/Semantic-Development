from langchain_community.document_loaders import TextLoader
# manually create an document
from langchain_core.documents import Document

docs = Document(
    page_content="this is an page content and know using for practice and then we will  move to the next loaders",
    metadata={
        'source':'information.txt',
        'author':'Hashir sameed',
        'page':1,
        'date':"21/11/2025"
    }
)
# text loader
from langchain_community.document_loaders import TextLoader
loader = TextLoader(
    r"C:/Users/hashi/Downloads/Semantic-Development-main/Semantic-Development-main/semantic search/rag/information.txt",
    encoding='utf-8'
)
load_loader= loader.load()
# embeddings Vector Db
import numpy as np 
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List , Dict  ,Any,Tuple
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingsManager:
    """ handles documents embeddings generation using sentence-transformer"""
    def __init__(self,model_name:str='all-MiniLM-L6-v2'):
        """
        initialized the embeddings manager
        args:
            model_name: hugging face model name for sentence embeddings
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """ load the sentencetransformer model """
        try:
            print(f"Loading embeddings model : {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"model loaded successfully. embeddings dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"error loading model {self.model_name}:{e}")
            raise
    def generate_embeddings(self , texts: list[str])->np.ndarray:
        """
        generate embeddings for a list of texts

        args:
            text:list of strings to embed
        return:
            numpy array of embeddings with shape (len(text),embaddings_dim)
        """
        if not self.model:
            raise ValueError("model not loaded")
        

        print(f"generate embeddings for {len(texts)}text")
        embeddings = self.model.encode(texts,show_progress_bar= True)
        print(f"generated embeddings with shape: {embeddings.shape}")
        return embeddings
    """def get_sentence_embedding_dimension(self)->int:
        # get the embedding dimension of the model
        if not self.model:
            raise ValueError("model not loaded")
        return self.model.get_sentence_embedding_dimension()"""
    
""" initialized the embeddings manager"""
embeddings_manager = EmbeddingsManager()
print(embeddings_manager)

#vector store
class VectorStore:
    def __init__(self,collection_name:str="pdf_document" , persist_directory:str=r"C:/Users/hashi/Downloads/Semantic-Development-main/Semantic-Development-main/semantic search/rag/information.txt"):
        """
        initialized the vector store
        args:
            collection name: name of the chromadb collection
            persist_directory: directory to persist vector store
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialized_store()
