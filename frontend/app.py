import streamlit as st
import requests
import pandas as pd
import io
import os
import math
import sys
import numpy as np
import datetime
import decimal

# Add the folder containing functions.py to the Python path
sys.path.append("/app/shared_utils")

# Import directly from the file
from functions import clean_sample_data, make_json_safe

BACKEND_URL = "http://query-agent-backend:8000"

st.set_page_config(page_title="Query Agent", layout="centered")
st.title("🧠 Query Agent")
st.caption("Ask questions in plain English and generate SQL queries using AI. Upload your data or connect to a database to get started.")

data_source = st.radio(
    "Choose your data source:", ["PostgreSQL Database", "Upload File"], horizontal=True
)
uploaded_file = None
user_df = None
selected_table = None

# Initialize tracker
if "prev_data_source" not in st.session_state:
    st.session_state.prev_data_source = data_source

# Detect a true change
if st.session_state.prev_data_source != data_source:
    # Clear session variables safely
    for key in ["generated_sql", "query_result_df", "modified_sql", "filters", "question_input"]:
        st.session_state.pop(key, None)

    # Also clear selected_table and uploaded file refs
    st.session_state.pop("selected_table", None)

    # Update the tracker
    st.session_state.prev_data_source = data_source

    # Reset selected table if data source changes
    selected_table = None

# --- PostgreSQL selection ---
if data_source == "PostgreSQL Database":
    try:
        tables_response = requests.get(f"{BACKEND_URL}/tables")
        tables = tables_response.json() if tables_response.ok else []
    except Exception as e:
        st.error(f"Failed to connect to backend: {e}")
        tables = []

    if not tables:
        st.warning("⚠️ No tables found in the database. Please add some tables and data to the PostgreSQL database.")
        st.stop()

    st.subheader("📋 Select a Table:")
    cols = st.columns(3)
    for i, table in enumerate(tables):
        if cols[i % 3].button(table):
            selected_table = table
            st.session_state["selected_table"] = table
            st.session_state.pop(f"data_dict_{table}", None)

    selected_table = st.session_state.get("selected_table", selected_table)

# --- File upload selection ---
elif data_source == "Upload File":
    selected_table = st.session_state.get("selected_table")
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        filename = uploaded_file.name

        # Check if new file has been uploaded
        if filename != st.session_state.get("uploaded_filename"):
            try:
                # Parse file
                if filename.endswith(".csv"):
                    user_df = pd.read_csv(uploaded_file)
                else:
                    user_df = pd.read_excel(uploaded_file)

                # Store file and dataframe in session
                st.session_state["uploaded_filename"] = filename
                st.session_state["uploaded_df"] = user_df

                # Prepare data for LLM
                sample_rows = make_json_safe(user_df.head(3).to_dict(orient="records"))

                # Auto-suggest table name via LLM
                response = requests.post(f"{BACKEND_URL}/suggest-table-name", json={
                    "columns": list(user_df.columns),
                    "sample_rows": sample_rows
                })

                if response.ok:
                    suggested_name = response.json()["suggested_name"]
                    st.session_state["selected_table"] = suggested_name
                    selected_table = suggested_name
                    st.info(f"🤖 Suggested Table Name: `{suggested_name}`")
                else:
                    st.warning("⚠️ Could not suggest table name.")
            except Exception as e:
                st.error(f"Failed to load file: {e}")
        else:
            # Load cached version on rerun
            user_df = st.session_state.get("uploaded_df")
            selected_table = st.session_state.get("selected_table")


# --- Data Dictionary and Sample Data ---
if selected_table:
    st.markdown(f"**Selected Table:** `{selected_table}`")

    # Cache keys
    dict_key = f"data_dict_{selected_table}" if selected_table else "data_dict_uploaded"
    sample_key = f"sample_data_{selected_table}" if selected_table else "sample_data_uploaded"

    # -----------------------
    # Sample Data
    # -----------------------
    if sample_key not in st.session_state:
        if data_source == "PostgreSQL Database":
            sample_data_res = requests.get(f"{BACKEND_URL}/sample-data/{selected_table}")
            if sample_data_res.ok:
                st.session_state[sample_key] = sample_data_res.json()
            else:
                st.warning("Could not retrieve sample data.")
                st.stop()
        elif data_source == "Upload File":
            # Send cleaned sample to backend to get diverse rows
            try:
                sample_rows = make_json_safe(user_df.head(500).to_dict(orient="records"))  # larger pool for diversity

                payload = {
                    "rows": sample_rows,
                    "n": 10
                }

                diverse_res = requests.post(f"{BACKEND_URL}/diverse-sample", json=payload)

                if diverse_res.ok:
                    st.session_state[sample_key] = diverse_res.json()["sample"]
                else:
                    st.warning("⚠️ Could not generate diverse sample. Using first 10 rows instead.")
                    st.session_state[sample_key] = user_df.head(10).to_dict(orient="records")

            except Exception as e:
                st.warning(f"⚠️ Failed to generate sample: {e}")
                st.session_state[sample_key] = user_df.head(10).to_dict(orient="records")

    # Display sample data
    sample_data = st.session_state[sample_key]

    if not sample_data:
        st.info("ℹ️ This data source is currently empty.")
    else:
        st.subheader("🔍 Sample Data")
        st.caption("These rows were selected to show diverse examples from your data. \
                    They help the LLM understand a broader range of patterns for better query generation."
                    )
        df_sample = pd.DataFrame(sample_data)

        edited_sample = st.data_editor(
            df_sample,
            use_container_width=True,
            hide_index=True,
            key="editable_sample"
            #num_rows="dynamic"
        )

        # Update session with edited sample data
        st.session_state[sample_key] = edited_sample.to_dict(orient="records")

    # -----------------------
    # Data Dictionary
    # -----------------------
    if dict_key not in st.session_state:
        if data_source == "PostgreSQL Database":
            dict_res = requests.post(f"{BACKEND_URL}/data-dictionary-postgres", json={
                "table_name": selected_table
            })
            if dict_res.ok:
                st.session_state[dict_key] = dict_res.json()["dictionary"]
            else:
                st.warning("Could not generate data dictionary.")
                st.stop()
        elif data_source == "Upload File":
            try:
                dict_res = requests.post(f"{BACKEND_URL}/data-dictionary-upload", json={
                    "table_name": selected_table,
                    "sample_data": make_json_safe(st.session_state[sample_key])
                })
                if dict_res.ok:
                    st.session_state[dict_key] = dict_res.json()["dictionary"]

                    st.json(dict_res.json())

                else:
                    st.warning("Could not generate data dictionary from file upload.")
                    st.stop()
            except Exception as e:
                st.error(f"Error generating dictionary from uploaded file: {e}")
                st.stop()

    st.subheader("📘 Data Dictionary")
    st.caption("Descriptions are generated using an LLM based on your table schema. You can modify them to guide query generation more precisely.")
    dictionary = st.session_state[dict_key]

    df_dict = pd.DataFrame(dictionary)
    edited_dict = st.data_editor(
        df_dict,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        key="data_dict_editor"
    )

    if st.button("🔄 Refresh Data Dictionary"):
        st.session_state.pop(dict_key, None)
        st.rerun()

# Question and SQL generation
if selected_table:
    question_key = f"sample_questions_{selected_table}"

    sample_data = clean_sample_data(sample_data)

    if question_key not in st.session_state:
        try:
            payload = {
                "table_name": selected_table,
                "question": "",
                "data_dictionary": edited_dict.to_dict(orient="records"),
                "sample_data": sample_data
            }
            res = requests.post(f"{BACKEND_URL}/generate-sample-questions", json=payload)

            if res.ok:
                all_questions = res.json()["questions"]
                st.session_state[question_key] = all_questions
            else:
                st.warning("Failed to get sample questions from backend.")
                st.session_state[question_key] = []
        except Exception as e:
            st.warning(f"⚠️ Sample question generation failed: {e}")
            st.session_state[question_key] = []

    # Show example questions if available
    example_qs = st.session_state.get(question_key, [])
    if example_qs:
        st.markdown("**💡 Example Questions:**")
        cols = st.columns(len(example_qs))
        for i, q in enumerate(example_qs):
            if cols[i].button(q, key=f"q_btn_{i}"):
                st.session_state["question_input"] = q
                st.rerun()

    # Question input field
    question = st.text_area("What do you want to know?", height=100, key="question_input")

    # Generate SQL button
    if st.button("Generate SQL"):
        payload = {
            "table_name": selected_table,
            "question": question,
            "data_dictionary": edited_dict.to_dict(orient="records"),
            "sample_data": sample_data
        }

        #st.write("📦 Payload being sent:", payload)

        res = requests.post(f"{BACKEND_URL}/generate-sql", json=payload)
        if res.ok:
            st.session_state["generated_sql"] = res.json()["sql"]
        else:
            st.error("Something went wrong!")

# Show Generated SQL
if "generated_sql" in st.session_state and selected_table:
    sql = st.session_state["generated_sql"]
    st.subheader("Generated SQL:")
    st.code(sql, language="sql")
    
    # Refresh dictionary button
    with st.expander("🔧 Refine Your Query"):
        filters = {}

        for row in edited_dict.to_dict(orient="records"):
            column = row["Column"]
            description = row["Description"]
            key = f"{selected_table}_{column}"
            use_key = f"use_{key}"

            sample_values = [r[column] for r in sample_data if column in r and r[column] is not None]
            unique_values = list(sorted(set(sample_values)))
            if not unique_values:
                continue

            # Checkbox to enable this field's filter
            use_filter = st.checkbox(f"Filter by {column}", key=use_key)

            # Categorical filter
            if all(isinstance(v, str) for v in unique_values):
                dropdown_key = f"{key}_dropdown"
                text_key = f"{key}_custom"

                options = ["Any"] + unique_values + ["Other"]

                # If the previous value was custom but not in sample, preserve it
                previous = st.session_state.get(dropdown_key)
                if previous not in options and previous is not None:
                    options.insert(-1, previous)  # insert before "Other"

                st.caption(f"{column} — {description}")

                selected = st.selectbox(
                    "Select value", 
                    options,
                    key=dropdown_key,
                    disabled=not use_filter
                )

                if use_filter and selected == "Other":
                    custom_value = st.text_input(
                        f"Enter custom value for {column}",
                        key=text_key,
                        disabled=not use_filter,
                        placeholder="Type value..."
                    )
                    if custom_value.strip():
                        filters[column] = f"{column} = '{custom_value.strip()}'"
                elif use_filter and selected != "Any":
                    filters[column] = f"{column} = '{selected}'"

            # Numeric filter (int or float)
            elif all(isinstance(v, (int, float)) for v in unique_values):
                min_val, max_val = min(unique_values), max(unique_values)
                min_key = f"{key}_min"
                max_key = f"{key}_max"

                st.session_state.setdefault(min_key, min_val)
                st.session_state.setdefault(max_key, max_val)

                st.caption(f"{column} — {description}")
                col1, col2 = st.columns(2)
                min_input = col1.number_input(
                    "Min",
                    key=min_key,
                    disabled=not use_filter,
                    min_value=min_val,
                    max_value=max_val
                )
                max_input = col2.number_input(
                    "Max",
                    key=max_key,
                    disabled=not use_filter,
                    min_value=min_val,
                    max_value=max_val
                )

                # Validation
                if use_filter:
                    if min_input > max_input:
                        st.warning(f"⚠️ `{column}`: Minimum value cannot exceed maximum.")
                    elif max_input < min_input:
                        st.warning(f"⚠️ `{column}`: Maximum value cannot be less than minimum.")
                    elif min_input != min_val or max_input != max_val:
                        filters[column] = f"{column} BETWEEN {min_input} AND {max_input}"

        # Button to manually generate modified SQL
        if st.button("Refine Query"):
            if filters:
                where_clause = " AND ".join(filters.values())
                modified_sql = f"SELECT * FROM {selected_table} WHERE {where_clause};"
            else:
                modified_sql = f"SELECT * FROM {selected_table};"

            st.session_state["modified_sql"] = modified_sql
            st.session_state["filters"] = filters

        # Show SQL only after it's generated
        if "modified_sql" in st.session_state:
            st.subheader("📝 Modified SQL")
            st.code(st.session_state["modified_sql"], language="sql")

            if st.button("▶️ Run Modified SQL"):
                run_res = requests.post(f"{BACKEND_URL}/run-sql", json={"sql": st.session_state["modified_sql"]})

                if run_res.ok:
                    result = run_res.json()
                    df_result = pd.DataFrame(result["rows"], columns=result["columns"])
                    st.session_state["query_result_df"] = df_result
                else:
                    st.error(f"Execution failed: {run_res.json().get('error', 'Unknown error')}")

    if st.button("▶️ Run Generated SQL"):
        run_res = requests.post(f"{BACKEND_URL}/run-sql", json={"sql": st.session_state["generated_sql"]})

        if run_res.ok:
            result = run_res.json()
            df_result = pd.DataFrame(result["rows"], columns=result["columns"])
            st.session_state["query_result_df"] = df_result
        else:
            st.error(f"Execution failed: {run_res.json().get('error', 'Unknown error')}")

# After query result is shown
if "query_result_df" in st.session_state and (selected_table or user_df is not None):
    df_result = st.session_state["query_result_df"]

    # Show the table
    st.subheader("📊 Query Results")
    st.dataframe(df_result)

    # Generate the summary
    try:
        summary_res = requests.post(f"{BACKEND_URL}/describe-results", json={
            "sql": st.session_state["generated_sql"],  # or modified_sql
            "rows": df_result.head(10).to_dict(orient="records")
        })

        if summary_res.ok:
            st.markdown("#### 🧠 Summary of Results")
            st.success(summary_res.json()["summary"])
        else:
            st.warning("Could not generate summary of results.")
    except Exception as e:
        st.warning(f"Error summarizing results: {e}")

    # Anomaly/Trend detection
    try:
        detect_res = requests.post(f"{BACKEND_URL}/detect-anomalies", json={
            "sql": st.session_state["generated_sql"],
            "rows": df_result.to_dict(orient="records")
        })

        if detect_res.ok:
            anomalies = detect_res.json().get("warnings", [])
            if anomalies:
                st.markdown("#### ⚠️ Anomalies / Warnings")
                with st.expander("View Anomaly Log", expanded=True):
                    log = "\n".join(f"- {w}" for w in anomalies)
                    st.code(log, language="markdown")
    except Exception as e:
        st.warning(f"Anomaly check failed: {e}")

    # Generate Chart
    try:
        chart_res = requests.post(f"{BACKEND_URL}/suggest-chart", json={
            "sql": st.session_state["generated_sql"],
            "rows": df_result.head(10).to_dict(orient="records")
        })

        if chart_res.ok:
            chart_info = chart_res.json()
            chart_type = chart_info["chart_type"]
            x = chart_info["x_axis"]
            y = chart_info["y_axis"]

            st.markdown("#### 📊 Suggested Chart")

            if not chart_type or chart_type == "none":
                st.info("ℹ️ No chart was recommended based on the query result. The data may not be suitable for visualization.")
            elif x not in df_result.columns or y not in df_result.columns:
                st.warning(f"⚠️ Suggested columns `{x}` or `{y}` not found in the query result.")
            else:
                st.write(f"**Type:** {chart_type.title()}  \n**X-axis:** `{x}`  \n**Y-axis:** `{y}`")

                chart_type = chart_info.get("chart_type", "").strip().lower()
                if chart_type == "bar_chart":
                    st.bar_chart(df_result.set_index(x)[y])
                elif chart_type == "line_chart":
                    st.line_chart(df_result.set_index(x)[y])
                elif chart_type == "area_chart":
                    st.area_chart(df_result.set_index(x)[y])
                elif chart_type == "scatter_chart":
                    st.scatter_chart(df_result[[x, y]])
                else:
                    st.warning(f"⚠️ Chart type `{chart_type}` is not supported for rendering.")

        else:
            st.warning("⚠️ Chart suggestion failed.")
    except Exception as e:
        st.warning(f"⚠️ Chart suggestion error: {e}")