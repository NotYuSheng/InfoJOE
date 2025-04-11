import streamlit as st
import requests
import pandas as pd
import time

# Add the folder containing functions.py to the Python path
sys.path.append("/app/shared_utils")

# Import directly from the file
from functions import clean_sample_data, make_json_safe

BACKEND_URL = "http://query-agent-backend:8000"

st.set_page_config(page_title="InfoJoe", layout="centered")
st.title("🧠 InfoJOE")
st.caption("Ask questions in plain English and generate SQL queries using AI. Upload your data or connect to a database to get started.")

user_df = None
selected_table = None

# --- PostgreSQL ---
with st.spinner("🔄 Connecting to Database..."):
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
        if cols[i % 3].button(table, use_container_width=True):
            selected_table = table
            st.session_state["selected_table"] = table
            st.session_state.pop(f"data_dict_{table}", None)

    selected_table = st.session_state.get("selected_table", selected_table)

# --- Data Dictionary and Sample Data ---
if selected_table:
    st.markdown(f"**Selected Table:** `{selected_table}`")

    # Cache keys
    dict_key = f"data_dict_{selected_table}" if selected_table else "data_dict_uploaded"
    sample_key = f"sample_data_{selected_table}" if selected_table else "sample_data_uploaded"

    # -----------------------
    # Sample Data
    # -----------------------
    with st.spinner("🔄 Generating Diverse Sample Data..."):
        if sample_key not in st.session_state:
            sample_data_res = requests.get(f"{BACKEND_URL}/sample-data/{selected_table}")
            if sample_data_res.ok:
                st.session_state[sample_key] = sample_data_res.json()
            else:
                st.warning("Could not retrieve sample data.")
                st.stop()

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
    with st.spinner("🔄 Generating Data Dictionary..."):
        if dict_key not in st.session_state:
            max_attempts = 3
            attempts = 0
            while attempts < max_attempts:
                attempts += 1
                dict_res = requests.post(f"{BACKEND_URL}/data-dictionary-postgres", json={
                    "table_name": selected_table,
                    "sample_data": make_json_safe(edited_sample.head(10).to_dict(orient="records"))
                })

                if dict_res.ok:
                    dictionary = dict_res.json().get("dictionary", [])
                    if dictionary:
                        st.session_state[dict_key] = dictionary
                        break
                    else:
                        st.info(f"⚠️ Data dictionary was empty. Retrying... (Attempt {attempts}/{max_attempts})")
                        time.sleep(1)  # Optional: wait before retrying
                else:
                    st.warning("❌ Failed to reach backend for data dictionary generation.")
                    st.stop()

            else:
                st.error("⚠️ Data dictionary could not be generated after multiple attempts.")
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

# --- Question and SQL generation ---
if selected_table:
    with st.spinner("🔄 Generating Sample Questions..."):
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

        res = requests.post(f"{BACKEND_URL}/generate-sql", json=payload)
        if res.ok:
            st.session_state["generated_sql"] = res.json()["sql"]
        else:
            st.error("Something went wrong!")

# --- Show Generated SQL ---
if "generated_sql" in st.session_state and selected_table:
    sql = st.session_state["generated_sql"]
    st.subheader("Generated SQL:")
    st.code(sql, language="sql")
    
    # Refresh dictionary button
    with st.expander("🔧 Customize Your Query"):
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
        with st.spinner("🔄 Running Generated SQL..."):
            run_res = requests.post(f"{BACKEND_URL}/run-sql", json={"sql": st.session_state["generated_sql"]})

            if run_res.ok:
                result = run_res.json()
                if result and "rows" in result and "columns" in result:
                    df_result = pd.DataFrame(result["rows"], columns=result["columns"])
                    st.session_state["query_result_df"] = df_result
                else:
                    st.error("⚠️ Unexpected response format from backend.")
            else:
                error_msg = run_res.json().get("error", "Unknown error")
                st.error(f"Execution failed: {error_msg}")

# --- Show Query Result ---
if "query_result_df" in st.session_state and (selected_table or user_df is not None):
    with st.spinner("🔄 Generating Summary Items..."):
        df_result = st.session_state["query_result_df"]
        cleaned_rows = make_json_safe(df_result.head(10).to_dict(orient="records"))

        # Show the table
        st.subheader("📊 Query Results")
        st.dataframe(df_result)

        # Generate the summary
        try:
            summary_res = requests.post(f"{BACKEND_URL}/describe-results", json={
                "sql": st.session_state["generated_sql"],  # or modified_sql
                "rows": cleaned_rows
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
                "rows": cleaned_rows
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
                "rows": cleaned_rows
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
