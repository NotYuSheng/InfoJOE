import streamlit as st
import requests

BACKEND_URL = "http://query-agent-backend:8000"

st.title("🧠 Query Agent")

# Get tables from backend
try:
    tables_response = requests.get(f"{BACKEND_URL}/tables")
    tables = tables_response.json() if tables_response.ok else []
except Exception as e:
    st.error(f"Failed to connect to backend: {e}")
    tables = []

if not tables:
    st.warning("⚠️ No tables found in the database. Please add some tables and data to the PostgreSQL database.")
    st.stop()

# Table selection
selected_table = None
st.subheader("Select a Table:")
cols = st.columns(3)

for i, table in enumerate(tables):
    if cols[i % 3].button(table):
        selected_table = table
        st.session_state["selected_table"] = table

# Use previously selected table if exists
selected_table = st.session_state.get("selected_table", selected_table)

if selected_table:
    st.markdown(f"**Selected Table:** `{selected_table}`")

    # Fetch and show data dictionary
    dict_res = requests.post(f"{BACKEND_URL}/data-dictionary", json={
        "table_name": selected_table
    })

    if dict_res.ok:
        st.subheader("📘 Data Dictionary (LLM-generated)")
        st.markdown(dict_res.json()["dictionary"])
    else:
        st.warning("Could not generate data dictionary.")

    # Sample data preview
    sample_data_res = requests.get(f"{BACKEND_URL}/sample-data/{selected_table}")

    if sample_data_res.ok:
        sample_data = sample_data_res.json()

        if not sample_data:
            st.info("ℹ️ This table exists but currently has no data.")
        else:
            st.subheader("🔍 Sample Data (Top 5 Rows)")
            st.dataframe(sample_data)
    else:
        st.warning("Could not retrieve sample data.")

# Question and SQL generation
if selected_table:
    st.markdown(f"**Selected Table:** `{selected_table}`")
    question = st.text_area("What do you want to know?", height=100)

    if st.button("Generate SQL"):
        res = requests.post(f"{BACKEND_URL}/generate-sql", json={
            "table_name": selected_table,
            "question": question
        })

        if res.ok:
            # st.subheader("Prompt Sent to LLM:")
            # st.code(res.json()["prompt"], language="text")
            st.subheader("Generated SQL:")
            st.code(res.json()["sql"], language="sql")
        else:
            st.error("Something went wrong!")
