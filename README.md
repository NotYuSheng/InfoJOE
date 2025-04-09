# 🤖 Query Agent

**Query Agent** is an AI-powered natural language interface for PostgreSQL databases. It allows users to explore and query structured data using plain English, powered by [Qwen 2.5](https://huggingface.co/Qwen/Qwen1.5-72B-Chat) running via [LM Studio](https://lmstudio.ai/).

Built with:
- 🐍 FastAPI backend
- 📈 Streamlit frontend
- 🐘 PostgreSQL database
- 🧠 LLM-powered SQL generation
- 🐳 Docker & Docker Compose

---

## 🚀 Features

- Auto-discovers tables in your PostgreSQL database
- Click-to-select table interface
- LLM-generated data dictionary for any table
- Preview top 5 sample rows from selected table
- Natural language-to-SQL translation using Qwen 2.5
- Fully containerized setup for local development

---

## 🧱 Project Structure

```
query-agent/
├── backend/            # FastAPI server
│   ├── main.py
│   ├── db.py
│   ├── prompt.py
│   └── requirements.txt
├── frontend/           # Streamlit interface
│   ├── streamlit_app.py
│   └── requirements.txt
├── postgres/           # Sample DB seed
│   └── init.sql
├── docker-compose.yml
└── README.md
```

---

## 🛠️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/notyusheng/query-agent.git
cd query-agent
```

### 2. Start the app

```bash
docker-compose up --build -d
```

- Frontend (Streamlit): [http://localhost:8080](http://localhost:8080)
- Backend (FastAPI): [http://localhost:3000/docs](http://localhost:3000/docs)
- Database (PostgreSQL): Port `5432` exposed locally

---

## 🧠 LLM Setup (Qwen 2.5 via LM Studio)

Make sure LM Studio is running locally or on your network with the following:
- Model: Qwen 2.5 (or any OpenAI-compatible LLM)
- Server URL: `http://192.168.1.130:1234/v1/chat/completions`

This is hardcoded in `backend/main.py`—update `LLM_URL` if needed.

---

## 🧰 Example Table Included

A sample `devices` table is loaded on first run via `postgres/init.sql`:

```sql
CREATE TABLE devices (
    device_id SERIAL PRIMARY KEY,
    model_name TEXT,
    os_version TEXT,
    release_year INTEGER
);
```

---

## 📸 Screenshots

_coming soon..._

---

## 📌 To-Do / Ideas

- [ ] Add support for multiple-table joins
- [ ] Semantic search across column descriptions
- [ ] Export SQL results as CSV
- [ ] Admin interface to manage metadata

---

## 📄 License

MIT License — free for personal and commercial use.

---

## 💬 Feedback / Contributions

Open an issue, submit a PR, or reach out if you want to collaborate. This is an ongoing project and evolving rapidly.

