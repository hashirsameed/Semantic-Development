from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

prompt = PromptTemplate(
    template="write a summary of this essay -\n {essay}",
    input_variables=['essay']
)
parser = StrOutputParser()
 
loader = TextLoader('rag/information.txt',encoding='utf-8')
docs = loader.load()
# print(docs)
# print(docs[0].page_content)
# print(docs[0].metadata)
chain = prompt | llm | parser
print(chain.invoke({'essay':docs[0].page_content}))