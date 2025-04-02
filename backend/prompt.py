def generate_sql_query(schema_rows, question, table_name):
    """
    Builds a prompt that provides the LLM with table schema and the user question.
    """
    schema_str = "\n".join([
        f"- {column} ({dtype})"
        for column, dtype in schema_rows
    ])

    prompt = f"""
You are a helpful assistant that generates SQL queries from natural language.

The user wants to query the `{table_name}` table.

Here is the schema of the `{table_name}` table:
{schema_str}

Based on the user's request, generate a syntactically correct SQL query.
Only use columns from the `{table_name}` table.

User's question:
\"\"\"{question}\"\"\"
"""
    return prompt.strip()

def generate_data_dictionary_prompt(table_name, schema_rows):
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