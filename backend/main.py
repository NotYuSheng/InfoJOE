from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from db import get_connection, get_table_schema, list_all_tables
from prompt import generate_sql_query, generate_data_dictionary_prompt
import requests
import pandas as pd

app = FastAPI()

# Allow frontend (Streamlit) to access the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LLM_URL = "http://192.168.1.130:1234/v1/chat/completions"

class QueryRequest(BaseModel):
    table_name: str
    question: str

@app.get("/tables")
def get_tables():
    return list_all_tables()

@app.post("/generate-sql")
def generate_sql(request: QueryRequest):
    schema = get_table_schema(request.table_name)
    prompt = generate_sql_query(schema, request.question, request.table_name)

    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "qwen2.5-7b-instruct-1m",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that generates SQL queries from natural language."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    response = requests.post(LLM_URL, headers=headers, json=payload)
    response.raise_for_status()
    sql = response.json()["choices"][0]["message"]["content"]

    return {
        "prompt": prompt,
        "sql": sql
    }

class DictionaryRequest(BaseModel):
    table_name: str

@app.post("/data-dictionary")
def get_data_dictionary(request: DictionaryRequest):
    schema = get_table_schema(request.table_name)
    prompt = generate_data_dictionary_prompt(request.table_name, schema)

    response = requests.post(LLM_URL, headers={"Content-Type": "application/json"}, json={
        "model": "qwen2.5-7b-instruct-1m",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that describes database tables."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    })
    response.raise_for_status()
    result = response.json()["choices"][0]["message"]["content"]
    return {"dictionary": result}

@app.get("/sample-data/{table_name}")
def get_sample_data(table_name: str):
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute(f"SELECT * FROM {table_name} LIMIT 5;")
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    finally:
        cur.close()
        conn.close()

    df = pd.DataFrame(rows, columns=columns)
    return df.to_dict(orient="records")
