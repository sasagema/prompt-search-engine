from pydantic import BaseModel
from promptSearchEngine import PromptSearchEngine
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import streamlit as st

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DATASET = "Gustavosta/Stable-Diffusion-Prompts"

class SearchRequest(BaseModel):
    query: str 
    n: int | None = 5
    
# model = SentenceTransformer("all-MiniLM-L6-v2")
# dataset = load_dataset("Gustavosta/Stable-Diffusion-Prompts" , split="test[:1%]")
# promptSearchEngine = PromptSearchEngine(dataset["Prompt"], model)

@st.cache_resource  
def load_model():
    """Initialize pretrained model for vectorizing.
        @st.cache_resource anotation enables caching for Streamlit.
    """
    return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_resource  
def load_dataSet():
    """Initialize pretrained model for vectorizing.
        @st.cache_resource anotation enables caching for Streamlit. 
    """
    return load_dataset(DATASET , split="test[:1%]")

@st.cache_resource  
def load_searchEngine(prompts, _model):
    """Initialize search engine and vectorize raw propmpts from dataset.
        @st.cache_resource anotation enables caching for Streamlit.
        Args:
        prompts: The sequence of raw prompts from the dataset.
        model: The model for vectorizing.
    """
    return PromptSearchEngine(prompts, _model)

model = load_model()
dataset = load_dataSet()
promptSearchEngine = load_searchEngine(dataset["Prompt"], model)


with st.form("search_form"):
    st.write("Prompt Search Engine")
    query = st.text_area("Prompt to search")
    number = st.number_input("Number of similar prompts", value = 5, min_value=0, max_value=100)
    submitted = st.form_submit_button("Submit")
    if submitted:
        result = promptSearchEngine.most_similar(query, number)
        st.dataframe(
            result, 
            use_container_width=True,
            column_config={
                1: st.column_config.NumberColumn(
                    "Similarity", 
                    help="Range in [-1, 1] where 1 is max similarity, means that prompts are identical.",
                    format= "%.4f"
                ),
                2: st.column_config.TextColumn("Prompts", help="The simlar prompts"),
            },
        )


