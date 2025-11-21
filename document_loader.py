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

### Text splitting get into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
def split_documents(documents,chunk_size=1000,chunk_overlap=200):
    """Split documents into smaller chunks for better RAG performance"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
    
    # Show example of a chunk
    if split_docs:
        print(f"\nExample chunk:")
        print(f"Content: {split_docs[0].page_content[:200]}...")
        print(f"Metadata: {split_docs[0].metadata}")
    
    return split_docs

chunks = split_documents
# embeddings Vector Db
import numpy as np 
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List , Dict  ,Any,Tuple
from sklearn.metrics.pairwise import cosine_similarity
import os

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
    def __init__(self,collection_name:str="information.txt" , persist_directory:str=r"C:/Users/hashi/Downloads/Semantic-Development-main/Semantic-Development-main/semantic search/rag"):
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

    def _initialized_store(self):
        """ initialized chromaDB client and collection"""
        try:
            #create persistent chromaDB client
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            #get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description":"text document embeddings for RAG"}
            )
            print(f"vector store initialized.collection: {self.collection_name}")
            print(f"existing document in collection : {self.collection.count()}")
        except Exception as e:
            print(f"error initializing vector store : {e}")
            raise

    def add_document(self,documents : list[Any], embeddings:np.ndarray):
        """
        add documents and thier embeddings to the vector DB

        ARGS:
            documents:list of langchain document
            embeddings:correspondings embeddings for the document
        """
        if len(documents) != len(embeddings):
            raise ValueError("number of documents must match the number of embeddings")
        
        print(f"adding {len(documents)} documents of vector store..")

        #prepare data for chromaDB
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []
        
        for i , (doc,embedding) in enumerate(zip(documents , embeddings)):
            # generate unique id
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            # Prepare metadata
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)
            
            # Document content
            documents_text.append(doc.page_content)
            
            # Embedding
            embeddings_list.append(embedding.tolist())
        # Add to collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )
            print(f"Successfully added {len(documents)} documents to vector store")
            print(f"Total documents in collection: {self.collection.count()}")
            
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise
vectorstore =VectorStore()
print(vectorstore)