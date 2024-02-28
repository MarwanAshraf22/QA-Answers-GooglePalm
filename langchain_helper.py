from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
load_dotenv()

from langchain.llms import GooglePalm
llm = GooglePalm(google_api_key=os.environ['GOOGLE_API_KEY'], temperature=0.7)

instructor_embedding = HuggingFaceEmbeddings()
vecdb_file = 'faiss_index'

def create_vectordb():
    loader = CSVLoader(file_path='codebasics_faqs.csv', source_column='prompt')
    docs = loader.load()
    vectordb = FAISS.from_documents(documents=docs, embedding=instructor_embedding)
    vectordb.save_local(vecdb_file)

def get_qa_chain():
    vector_db = FAISS.load_local(vecdb_file,instructor_embedding)
    retriever = vector_db.as_retriever(source_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.
    CONTEXT: {context}
    QUESTION: {question}"""

    prompt = PromptTemplate(
        template=prompt_template, input_variables=['context', 'question']
    )
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever,
                                        input_key='query', return_source_documents=True,
                                        chain_type_kwargs={'prompt':prompt})
    return chain


if __name__ == '__main__':
    chain = get_qa_chain()
    print(chain('Do you have internships'))




