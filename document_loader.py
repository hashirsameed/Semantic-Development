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

loader=TextLoader("information.text",encoding="utf-8")
load_loader= loader.load()
# embeddings Vector Db
import numpy as np 
from sentence_transformers import sentencetransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List , Dict  ,Any,Tuple
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingsManager:
    """ handles documents embaddings generation using sentence-transformer"""
    def __init__(self,model_name:str='all-MiniLM-L6-v2'):
        """
        initialized the embaddings manager
        args:
            model_name: hugging face model name for sentence embaddings
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """ load the sentencetransformer model """
        try:
            print(f"Loading embaddings model : {self.model_name}")
            self.model = sentencetransformer(self.model_name)
            print(f"model loaded successfully. embaddings dimension: {self.model.get_sentence_embadding_dimension()}")
        except Exception as e:
            print(f"error loading model {self.model_name}:{e}")
            raise
    def generate_embaddings(self , texts: list[str])->np.ndarray:
        """
        generate embaddings for a list of texts

        args:
            text:list of strings to embed
        return:
            numpy array of embaddings with shape (len(text),embaddings_dim)
        """
        if not self.model:
            raise ValueError("model not loaded")
        

        print(f"generate embaddings for {len(texts)}text")
        embaddings = self.model.encode(texts,show_progress_bar= True)
        print(f"generated embaddings with shape: {embaddings.shape}")
        return embaddings
    def get_sentence_embadding_dimension(self)->int:
        """ get the embadding dimension of the model"""
        if not self.model:
            raise ValueError("model not loaded")
        return self.model.get_sentence_embadding_dimension()
    