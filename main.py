import streamlit as st
from langchain_helper import get_qa_chain,create_vectordb

st.title('QA from CSV ❓')

question = st.text_input('Question:')

if question :
    chain = get_qa_chain()
    response = chain(question)
    st.header("Answer : ")
    st.write(response['result'])