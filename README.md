
# Prompt Search Engine

Search engine that matches and suggests the best
prompts for Stable Diffusion models based on an initial input. The goal is to improve the quality
of image generation by providing more effective and relevant prompts.




## Dataset

Dataset of prompts for Stable Diffusion models used in this project.

https://huggingface.co/datasets/Gustavosta/Stable-Diffusion-Prompts


## Prompt Vectorizer

Transformation from textual prompt to numerical vectors is done with pre trained Sentence Transformers model "all-MiniLM-L6-v2" from HugginFace organization.
This model is trained on all available training data (more than 1 billion training pairs) and is designed as a general purpose model.
## Installation

Project is implemented using python=3.12.4 version

- Create virtual environment

```bash
python -m venv .env
```
Mac OS

```bash
source .venv/bin/activate
```
Windows PowerShell
```bash
.venv\Scripts\Activate.ps1
```

Install requirements
```bash
pip install -r requirements.txt
```

## Search API 

Implemented using fastapi.
To run fastapi API local server

```bash
  uvicorn run:app --reload --port=8000 --host=0.0.0.0
```
- API will be availiable on http://localhost:8000/
- API Documentation: http://localhost:8000/docs

### API Endpoints

    - GET http://localhost:8000/search?q={string}&n={int}

    q = Required
    n = Optional, default = 5



    - POST http://localhost:8000/search

    Body:

        {
            "query": {string} (Required), 
            "n": {int} (Optional, default value = 5) 
        }


Search API is deployed to Hugging Face Spaces:

https://sasagema-prompt-search-engine.hf.space/

Swagger documentation:

https://sasagema-prompt-search-engine.hf.space/docs


##  User Interface

Implemented using Streamlit
```bash
  streamlit run run_ui.py
```
  Local URL: http://localhost:8501

 User Interface is depoloyed to Hugging Face Spaces:

  https://sasagema-prompt-search-engine-ui.hf.space

## Docker image

Build:
```bash
  docker compose up --build
```

Run:
```bash
  docker run -p 8000:8000 prompt-search-engine
```


 
## How to Use

### Local version
- Run Search API
- Run User Interface
- Go to Local URL: http://localhost:8501

### Deployed

- "Wake up" Search API:

Go To: https://sasagema-prompt-search-engine.hf.space/docs

- User Interface:

Go To: https://sasagema-prompt-search-engine-ui.hf.space



