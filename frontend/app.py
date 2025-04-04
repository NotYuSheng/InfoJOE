import streamlit as st
import requests
import pandas as pd
import io

BACKEND_URL = "http://query-agent-backend:8000"

st.set_page_config(page_title="Query Agent", layout="centered")
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
        # Clear cached dictionary if switching tables
        st.session_state.pop(f"data_dict_{table}", None)

# Use previously selected table if exists
selected_table = st.session_state.get("selected_table", selected_table)

if selected_table:
    st.markdown(f"**Selected Table:** `{selected_table}`")

    # Data dictionary key
    dict_key = f"data_dict_{selected_table}"

    # Fetch and cache data dictionary only once
    if dict_key not in st.session_state:
        dict_res = requests.post(f"{BACKEND_URL}/data-dictionary", json={
            "table_name": selected_table
        })

        if dict_res.ok:
            st.session_state[dict_key] = dict_res.json()["dictionary"]
        else:
            st.warning("Could not generate data dictionary.")
            st.stop()

    # Show editable dictionary
    st.subheader("📘 Data Dictionary (LLM-generated)")
    dictionary = st.session_state[dict_key]

    df = pd.DataFrame(dictionary)
    edited_dict = st.data_editor(
        df,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        key="data_dict_editor"
    )

    if st.button("🔄 Refresh Data Dictionary"):
        st.session_state.pop(dict_key, None)  # remove cached dictionary
        st.rerun()  # triggers a full rerun

    # Sample data preview
    sample_key = f"sample_data_{selected_table}"

    # Load sample data once per table
    if sample_key not in st.session_state:
        sample_data_res = requests.get(f"{BACKEND_URL}/sample-data/{selected_table}")
        if sample_data_res.ok:
            st.session_state[sample_key] = sample_data_res.json()
        else:
            st.warning("Could not retrieve sample data.")
            st.stop()

    sample_data = st.session_state[sample_key]

    if not sample_data:
        st.info("ℹ️ This table exists but currently has no data.")
    else:
        st.subheader("🔍 Sample Data (10 Diverse Rows)")
        df = pd.DataFrame(sample_data)
        st.dataframe(df, height=200)

# Question and SQL generation
if selected_table:
    st.markdown(f"**Selected Table:** `{selected_table}`")

    # Generate sample questions once per table selection
    question_key = f"sample_questions_{selected_table}"

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
                st.session_state[question_key] = []
        except Exception as e:
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

if "generated_sql" in st.session_state:
    sql = st.session_state["generated_sql"]
    st.subheader("Generated SQL:")
    st.code(sql, language="sql")
    
    # Optional: Refresh dictionary button
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

                selected = st.selectbox(
                    f"{column} ({description})", options,
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

            # Numeric filter (min/max)
            elif all(isinstance(v, int) for v in unique_values):
                min_val, max_val = min(unique_values), max(unique_values)
                min_key = f"{key}_min"
                max_key = f"{key}_max"

                if min_key not in st.session_state:
                    st.session_state[min_key] = min_val
                if max_key not in st.session_state:
                    st.session_state[max_key] = max_val

                col1, col2 = st.columns(2)
                min_input = col1.number_input(
                    f"Min {column} ({description})",
                    key=min_key,
                    disabled=not use_filter
                )
                max_input = col2.number_input(
                    f"Max {column} ({description})",
                    key=max_key,
                    disabled=not use_filter
                )

                if use_filter and (min_input != min_val or max_input != max_val):
                    filters[column] = f"{column} BETWEEN {int(min_input)} AND {int(max_input)}"

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
if "query_result_df" in st.session_state:
    df_result = st.session_state["query_result_df"]

    # show the table
    st.subheader("📊 Query Results")
    st.dataframe(df_result)

    # generate the summary
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

