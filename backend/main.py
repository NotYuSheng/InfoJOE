from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from db import get_connection, get_table_schema, list_all_tables
from prompt import generate_sql_query, generate_data_dictionary_prompt
import requests
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import pairwise_distances
import numpy as np
import re

app = FastAPI()

# Allow frontend (Streamlit) to access the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LLM_URL = "http://192.168.1.142:1234/v1/chat/completions"

def get_diverse_sample(df: pd.DataFrame, n=10):
    if len(df) <= n:
        return df  # not enough rows to sample, return all

    df_copy = df.copy()
    df_copy.fillna("N/A", inplace=True)

    for col in df_copy.columns:
        if df_copy[col].dtype == "object":
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col].astype(str))

    distance_matrix = pairwise_distances(df_copy, metric='euclidean')
    selected_indices = [np.random.randint(len(df_copy))]

    for _ in range(n - 1):
        remaining = list(set(range(len(df_copy))) - set(selected_indices))
        if not remaining:
            break
        max_distances = distance_matrix[remaining][:, selected_indices].mean(axis=1)
        next_index = remaining[np.argmax(max_distances)]
        selected_indices.append(next_index)

    return df.iloc[selected_indices]

class QueryRequest(BaseModel):
    table_name: str
    question: str

@app.get("/tables")
def get_tables():
    return list_all_tables()

class GenerateSQLRequest(BaseModel):
    table_name: str
    question: str
    data_dictionary: list  # List of dicts: [{"Column": ..., "Description": ...}]
    sample_data: list      # List of sample rows as dicts

@app.post("/generate-sql")
def generate_sql(req: GenerateSQLRequest):
    dict_lines = [f"- {col['Column']}: {col['Description']}" for col in req.data_dictionary]
    dict_section = "\n".join(dict_lines)

    df_sample = pd.DataFrame(req.sample_data)
    diverse_sample = get_diverse_sample(df_sample, n=10)
    sample_lines = [str(row) for _, row in diverse_sample.iterrows()]
    sample_section = "\n".join(sample_lines)

    prompt = f"""
    You are an assistant that generates SQL queries from natural language.

    The user is asking a question about the `{req.table_name}` table.

    ### Data Dictionary:
    {dict_section}

    ### Sample Data:
    {sample_section}

    User's question:
    \"\"\"{req.question}\"\"\"

    Generate a SQL SELECT query that best answers the question.
    Use only the `{req.table_name}` table. Do not return explanations—only the SQL.
    """

    response = requests.post(LLM_URL, headers={"Content-Type": "application/json"}, json={
        "model": "qwen2.5-7b-instruct-1m",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that generates SQL queries from natural language."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    })
    response.raise_for_status()

    # Clean the raw SQL
    sql_raw = response.json()["choices"][0]["message"]["content"]
    sql_clean = sql_raw.strip()

    if sql_clean.startswith("```"):
        lines = sql_clean.splitlines()
        lines = [line for line in lines if not line.strip().startswith("```")]
        sql_clean = "\n".join(lines).strip()

    return {"sql": sql_clean, "prompt": prompt}


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
    print("LLM Response:\n", result)  # <- Debugging line

    entries = []
    for line in result.strip().splitlines():
        match = re.match(r"-\s*`?(\w+)`?\s*:\s*(.+)", line)
        if match:
            col, desc = match.groups()
            entries.append({
                "Column": col,
                "Description": desc
            })

    return {"dictionary": entries}

@app.get("/sample-data/{table_name}")
def get_sample_data(table_name: str):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(f"SELECT * FROM {table_name} LIMIT 100;")
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    finally:
        cur.close()
        conn.close()

    df = pd.DataFrame(rows, columns=columns)

    # Apply diverse sampling (10 rows)
    if len(df) > 10:
        df = get_diverse_sample(df, n=10)

    return df.to_dict(orient="records")

class RunSQLRequest(BaseModel):
    sql: str

@app.post("/run-sql")
def run_sql(request: RunSQLRequest):
    sql = request.sql.strip().lower()

    FORBIDDEN = ["drop", "delete", "insert", "update", "alter", "truncate"]
    if any(word in sql for word in FORBIDDEN):
        return JSONResponse(
            content={"error": "Query contains forbidden keywords."},
            status_code=403
        )

    # Restrict to SELECT statements only
    if not sql.startswith("select"):
        return JSONResponse(
            content={"error": "Only SELECT statements are allowed."},
            status_code=403
        )

    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(request.sql)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        cur.close()
        conn.close()
        return {"columns": columns, "rows": rows}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
