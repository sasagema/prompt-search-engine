from pydantic import BaseModel
from promptSearchEngine import PromptSearchEngine
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import streamlit as st
import requests
import json

st.title('Prompt Search Engine')

with st.form("search_form"):
    st.write("Prompt Search Engine")
    query = st.text_area("Prompt to search")
    number = st.number_input("Number of similar prompts", value = 5, min_value=0, max_value=100)
    submitted = st.form_submit_button("Submit")
    if submitted:
        inputs = {"query": query, "n": number}
        result = requests.post(url = "http://localhost:8000/search", data = json.dumps(inputs))
        result = result.json()
        st.dataframe(
            result["data"], 
            use_container_width=True,
            column_config={
                "similarity": st.column_config.NumberColumn(
                    "Similarity", 
                    help="Range in [-1, 1] where 1 is max similarity, means that prompts are identical.",
                    format= "%.4f"
                ),
                "prompt": st.column_config.TextColumn("Prompts", help="The simlar prompts"),
            },
        )


