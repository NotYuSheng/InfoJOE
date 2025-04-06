def create_sql_query_prompt(schema_rows, question, table_name):
    """
    Builds a prompt that provides the LLM with table schema and the user question.
    """
    schema_str = "\n".join([
        f"- {column} ({dtype})"
        for column, dtype in schema_rows
    ])

    prompt = f"""
    You are a SQL assistant. Your job is to generate safe, read-only SQL queries using only the `SELECT` statement.

    The user wants to query the `{table_name}` table.

    Here is the schema of the `{table_name}` table:
    {schema_str}

    IMPORTANT RULES:
    - You must ONLY use `SELECT` statements.
    - Do NOT use INSERT, UPDATE, DELETE, DROP, TRUNCATE, or any DDL/DML commands.
    - Only use columns from the `{table_name}` table.
    - Do NOT include any explanatory text—only return the raw SQL query.

    User's question:
    \"\"\"{question}\"\"\"
    """
    return prompt.strip()

def create_data_dictionary_prompt(table_name, schema_rows):
    schema_str = "\n".join([
        f"- {column} ({dtype})"
        for column, dtype in schema_rows
    ])
    return f"""
    Given the schema of the `{table_name}` table:

    {schema_str}

    Generate a data dictionary that describes what each column likely means in plain English.
    Respond in a markdown list format, one item per column.
    """

def create_data_dictionary_prompt_from_sample_data(table_name: str, sample_data: list[dict]) -> str:
    # Add clear instructions for the LLM to generate a data dictionary
    prompt = f"""
    You are an expert data analyst. Please describe the purpose of each column in the table named '{table_name}'.
    Based on the following sample data, provide a description for each column. The sample data includes the first few rows:

    {sample_data}

    For each column, provide a short description of its meaning and its role in the dataset. Here's an example format:

    - `column_name`: A description of the column

    Please ensure the descriptions are short and clear.

    Thank you!
    """
    return prompt
