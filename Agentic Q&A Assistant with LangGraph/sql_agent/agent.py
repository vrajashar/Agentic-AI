from langchain_groq import ChatGroq
from sql_agent.prompts import build_prompt
from sql_agent.validation import validate_sql
from sql_agent.db import run_query
from dotenv import load_dotenv

load_dotenv(override=True)

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

def generate_and_execute(question: str):
    prompt = build_prompt(question)
    response = llm.invoke(prompt)

    sql = response.content.strip()
    validate_sql(sql)

    result = run_query(sql)

    return sql, result
