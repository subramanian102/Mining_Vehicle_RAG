import os
from dotenv import load_dotenv
import torch
import transformers
from transformers import AutoTokenizer
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from a .env file
load_dotenv()

class LLMModel:
    def __init__(self, model_type="gemini"):
        self.model_type = model_type
        self.pipeline = None
        self.tokenizer = None
        self.gemini_model = None

        if self.model_type == "llama":
            model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
            self.pipeline, self.tokenizer = self.initialize_llama_model(model_name)
        elif self.model_type == "gemini":
            self.gemini_model = self.initialize_gemini_model()

    def initialize_llama_model(self, model_name):
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            raise ValueError("HF_TOKEN missing from environment variables.")
            
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=hf_token
        )
        return pipeline, tokenizer

    def initialize_gemini_model(self):
        google_api_key = os.getenv('GOOGLE_API_KEY')
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY missing from environment variables.")
            
        # ❌ OLD DEPRECATED MODEL CODE:
        # chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)
        
        #  NEW ACTIVE STABLE MODEL CODE:
        chat_model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            google_api_key=google_api_key
        )
        return chat_model

def get_rag_chain(persist_dir="chroma_db", model_type="gemini"):
    """Loads the database and hooks it into the RAG pipeline wrapper"""
    # Load embedding function
    embedding_function = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Load existing database
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(f"Database folder '{persist_dir}' not found. Run ingestion first.")
        
    db = Chroma(persist_directory=persist_dir, embedding_function=embedding_function)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    # Initialize LLM wrapper
    llm_instance = LLMModel(model_type=model_type)
    
    # Define system template
    template = """Answer the question based only on the following context derived from heavy mining machinery manuals. 
    If you do not know the answer based on the context, say gracefully that you cannot find it in the manual.
    
    Context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    # Build chain
    if model_type == "gemini":
        llm_engine = llm_instance.gemini_model
    else:
        # Fallback raw model execution if llama pipeline is chosen
        llm_engine = llm_instance.pipeline 

    retreival_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm_engine
        | StrOutputParser()
    )
    
    return retreival_chain