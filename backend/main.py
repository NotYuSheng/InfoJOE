from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from db import get_connection, get_table_schema, list_all_tables
from prompt import create_data_dictionary_prompt, create_data_dictionary_prompt_from_sample_data
import requests
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import pairwise_distances
import numpy as np
import re
import sys
from typing import List, Dict
import math
import datetime
import decimal

# Add the folder containing functions.py to the Python path
sys.path.append("/app/shared_utils")

# Import directly from the file
from functions import clean_sample_data, make_json_safe

app = FastAPI()

# Allow frontend (Streamlit) to access the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LLM_URL = "http://192.168.1.16:1234/v1/chat/completions"

def get_diverse_sample(df: pd.DataFrame, n=10) -> pd.DataFrame:
    if len(df) <= n:
        return df

    # Clean and encode
    df_copy = df.copy()

    # Step 1: Fill missing values safely
    for col in df_copy.columns:
        if df_copy[col].dtype == "object":
            df_copy.loc[:, col] = df_copy[col].fillna("N/A")
        else:
            df_copy.loc[:, col] = df_copy[col].fillna(0)

    # Step 2: Encode object columns using LabelEncoder
    for col in df_copy.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df_copy.loc[:, col] = le.fit_transform(df_copy[col].astype(str))

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

class PostgresDictionaryRequest(BaseModel):
    table_name: str

@app.post("/data-dictionary-postgres")
def generate_postgres_data_dictionary(request: PostgresDictionaryRequest):
    schema = get_table_schema(request.table_name)
    prompt = create_data_dictionary_prompt(request.table_name, schema)

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

class UploadDictionaryRequest(BaseModel):
    table_name: str
    sample_data: list[dict]

@app.post("/data-dictionary-upload")
def generate_upload_data_dictionary(request: UploadDictionaryRequest):
    # Create the prompt using the sample data
    prompt = create_data_dictionary_prompt_from_sample_data(
        request.table_name,
        request.sample_data
    )

    # Send the request to the LLM
    response = requests.post(LLM_URL, headers={"Content-Type": "application/json"}, json={
        "model": "qwen2.5-7b-instruct-1m",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that describes columns based on sample data."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    })

    response.raise_for_status()
    result = response.json()["choices"][0]["message"]["content"]

    # Parse the result into column descriptions
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

    if len(df) > 10:
        df = get_diverse_sample(df, n=10)

    # Convert and sanitize before returning
    raw_dicts = df.to_dict(orient="records")
    safe_dicts = clean_sample_data(raw_dicts)
    
    return safe_dicts

class SampleRequest(BaseModel):
    rows: list[dict]
    n: int = 10

@app.post("/diverse-sample")
def get_diverse_sample_endpoint(req: SampleRequest):
    df = pd.DataFrame(req.rows)

    diverse_df = get_diverse_sample(df, n=req.n)

    # Convert to JSON-safe structure
    sample_rows = diverse_df.to_dict(orient="records")

    # Clean it thoroughly (np.int64, datetime, etc.)
    return {"sample": clean_sample_data(sample_rows)}

class RunSQLRequest(BaseModel):
    sql: str

@app.post("/run-sql")
def run_sql(request: RunSQLRequest):
    sql = request.sql.strip().lower()

    FORBIDDEN = ["drop", "delete", "insert", "update", "alter", "truncate"]
    forbidden_word = next((word for word in FORBIDDEN if word in sql.lower()), None)

    if forbidden_word:
        return JSONResponse(
            content={"error": f"Query contains forbidden keyword: '{forbidden_word}'"},
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

@app.post("/generate-sample-questions")
def generate_sample_questions(req: GenerateSQLRequest):
    dict_lines = [f"- {col['Column']}: {col['Description']}" for col in req.data_dictionary]
    dict_section = "\n".join(dict_lines)

    df_sample = pd.DataFrame(req.sample_data)
    if not df_sample.empty:
        diverse_sample = get_diverse_sample(df_sample, n=10)
        sample_lines = [str(row) for _, row in diverse_sample.iterrows()]
        sample_section = "\n".join(sample_lines)
    else:
        sample_section = "(No sample data provided)"

    # prompt = f"""
    # You are a helpful assistant that suggests example questions users might ask about the {req.table_name} table.

    # ### Data Dictionary:
    # {dict_section}

    # ### Sample Data:
    # {sample_section}

    # Generate 3 example natural language questions that could be answered using a SQL SELECT query on the {req.table_name} table.
    # Do not include any explanations—only the questions, each as a separate bullet point.
    # """

    prompt = f"""
    You are a helpful assistant supporting analysts in counter-terrorism intelligence gathering.

    Given the structure and sample data of the `{req.table_name}` table, generate example investigative questions that an analyst might ask to uncover patterns, threats, or anomalies from the data.

    Questions should be practical, focused, and answerable using a SQL SELECT query.

    ### Data Dictionary:
    {dict_section}

    ### Sample Data:
    {sample_section}

    Generate 3 example investigative questions that could be answered using a SQL SELECT query on the `{req.table_name}` table.
    Do not include any explanations—only the questions, each as a separate bullet point.
    """

    response = requests.post(LLM_URL, headers={"Content-Type": "application/json"}, json={
        "model": "qwen2.5-7b-instruct-1m",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that suggests natural language queries for SQL generation."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    })
    response.raise_for_status()

    raw_text = response.json()["choices"][0]["message"]["content"]
    questions = [q.strip("•- ").strip() for q in raw_text.strip().splitlines() if q.strip()]
    return {"questions": questions}

class DescribeResultsRequest(BaseModel):
    sql: str
    rows: List[Dict]

@app.post("/describe-results")
def describe_results(req: DescribeResultsRequest):
    df = pd.DataFrame(req.rows)
    sample_lines = df.head(10).to_string(index=False)

    prompt = f"""
    You are a data analyst assistant.

    Given the following SQL query and sample results from that query, provide a short natural language summary of what the query result is showing. Do not explain SQL — just describe the result as if speaking to a non-technical user.

    ### SQL Query:
    {req.sql}

    ### Sample Results (top 10 rows):
    {sample_lines}

    Summarize the data shown above in 1–2 sentences.
    """

    response = requests.post(LLM_URL, headers={"Content-Type": "application/json"}, json={
        "model": "qwen2.5-7b-instruct-1m",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant who explains SQL query results to business users."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    })
    response.raise_for_status()

    summary = response.json()["choices"][0]["message"]["content"].strip()
    return {"summary": summary}

@app.post("/suggest-chart")
def suggest_chart(req: DescribeResultsRequest):
    df = pd.DataFrame(req.rows)
    sample_lines = df.head(10).to_string(index=False)

    prompt = f"""
    You are a data visualization assistant.

    Given the following SQL query and sample result data, decide whether a chart is useful.

    If a chart makes sense, suggest:
    - One of the following supported chart types:
    - area_chart
    - bar_chart
    - line_chart
    - scatter_chart
    - scatter_map (for geographical data like lat/lon)

    Also suggest:
    - A column for the X-axis
    - A column for the Y-axis

    If a chart does **not** make sense (e.g., text-heavy, too few rows, non-numeric data), say:
    Chart Type: none

    ### SQL Query:
    {req.sql}

    ### Sample Results (top 10 rows):
    {sample_lines}

    Respond in this exact format:
    Chart Type: <area_chart, bar_chart, line_chart, scatter_chart, scatter_map, none>
    X-Axis: <column name or 'None'>
    Y-Axis: <column name or 'None'>
    """

    response = requests.post(LLM_URL, headers={"Content-Type": "application/json"}, json={
        "model": "qwen2.5-7b-instruct-1m",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant for visualizing SQL result data."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    })
    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]

    # Parse response into parts
    lines = content.strip().splitlines()
    chart_type = next((line.split(":")[1].strip().lower() for line in lines if line.lower().startswith("chart type")), "none")
    x_axis = next((line.split(":")[1].strip() for line in lines if line.lower().startswith("x-axis")), None)
    y_axis = next((line.split(":")[1].strip() for line in lines if line.lower().startswith("y-axis")), None)

    # Normalize 'none'
    if x_axis and x_axis.lower() == "none":
        x_axis = None
    if y_axis and y_axis.lower() == "none":
        y_axis = None

    return {
        "chart_type": chart_type,
        "x_axis": x_axis,
        "y_axis": y_axis
    }

@app.post("/detect-anomalies")
def detect_anomalies(req: DescribeResultsRequest):
    df = pd.DataFrame(req.rows)

    if df.empty:
        return {"warnings": ["Query returned no data. You may want to revise your filters or try a broader question."]}

    warnings = []

    # Too few rows
    if len(df) < 5:
        warnings.append(f"Query returned only {len(df)} rows — consider broadening your filter.")

    # Check for dominant values
    for col in df.columns:
        top_freq = df[col].value_counts(normalize=True).max()
        if top_freq >= 0.9:
            dominant_val = df[col].value_counts().idxmax()
            warnings.append(f"Column `{col}` is dominated by `{dominant_val}` ({int(top_freq * 100)}%).")

    # Check if all values in a column are the same
    for col in df.columns:
        if df[col].nunique() == 1:
            val = df[col].iloc[0]
            warnings.append(f"All values in `{col}` are `{val}` — is this intentional?")

    return {"warnings": warnings}

class TableNameRequest(BaseModel):
    columns: list[str]
    sample_rows: list[dict]

@app.post("/suggest-table-name")
def suggest_table_name(req: TableNameRequest):
    prompt = (
        "Suggest a short, SQL-safe, lowercase table name for the following dataset.\n"
        "Columns: " + ", ".join(req.columns) + "\n\n"
        "Sample rows:\n" + "\n".join(str(row) for row in req.sample_rows[:3]) + "\n\n"
        "Respond with only the suggested table name (no explanations)."
    )

    response = requests.post(LLM_URL, json={
        "model": "qwen2.5-7b-instruct-1m",
        "messages": [
            {"role": "system", "content": "You are an expert data modeler."},
            {"role": "user", "content": prompt}
        ]
    })

    response.raise_for_status()
    name = response.json()["choices"][0]["message"]["content"]
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name.strip())  # ensure SQL-safe
    return {"suggested_name": name.lower()}
