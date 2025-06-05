from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_core.documents import Document
from typing import List
from langchain_core.messages import SystemMessage
from langchain.prompts import ChatPromptTemplate,HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

class NewsSummarizer:
    """This class summarizes news from URLs provided by the user"""
    def __init__(self):
        self.llm=ChatOllama(model="mistral",temperature=0.5)
        self.embed_fn=OllamaEmbeddings(model="nomic-embed-text")
    
    def load_news(self,urls:List[str])->List[Document]:
        """Loads the content of the URLs and returns a List[Documents]"""
        loader=UnstructuredURLLoader(urls=urls)
        docs=loader.load()
        return docs
    
    def create_chunks(self,docs:List[Document])->List[Document]:
        """Splits the document into several chunks for indexing them"""
        splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=200
                )
        chunks = splitter.split_documents(docs)
        return chunks
    
    def create_vectordb(self,chunks:List[Document])->FAISS:
        """Create a vector indexer using FAISS"""
        vectordb = FAISS.from_documents(
                    documents=chunks, 
                    embedding=self.embed_fn
                )
        return vectordb
    
    def retrieve_query(self, query: str, vectordb: FAISS) -> tuple[str, List[str]]:
        """Retrieves the relevant chunks and their sources."""
        retrieved_docs = vectordb.similarity_search(query, k=4)
        if not retrieved_docs:
            return "Not enough information found in the retrieved articles to answer confidently.", []
    
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        sources = list({doc.metadata.get("source", "Unknown Source") for doc in retrieved_docs})
        return context, sources

    
    def SummaryRAG(self,query:str,context:str)->str:
        """This function uses RAG to summarize the news based on the context retrived from the vector database"""
        sys_prompt = SystemMessage(
            content="""You are a professional financial assistant. Based on the retrieved news articles, generate a concise, bullet-point summary focused on the stock mentioned in the user's query. Include only relevant insights and remove any unrelated information. 

                        Your output should:
                        - Use clear, factual bullet points.
                        - Focus on recent news, financial performance, partnerships, or legal issues.
                        - Highlight anything that may influence the stock price positively or negatively.
                        - Avoid speculation or personal opinion.

                        If the context is insufficient, reply with: "Not enough information found in the retrieved articles to answer confidently."
                    """ 
        )
        
        human_msg = HumanMessagePromptTemplate.from_template(
            "Answer the following based on context:\n\nQuestion: {query}\n\nContext:\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([sys_prompt, human_msg])
        summary_chain= prompt | self.llm | StrOutputParser()

        result=summary_chain.invoke({"query": query, "context": context})
        return result