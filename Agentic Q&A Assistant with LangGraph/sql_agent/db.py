from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv(override=True)
DB_URI = os.getenv("DB_URI") 
if not DB_URI:
    raise ValueError("DB_URI environment variable is not set.")

langchain_db = SQLDatabase.from_uri(DB_URI)
engine = create_engine(DB_URI)

def get_langchain_db():
    return langchain_db

def get_schema() -> str:
    """
    Returns clean table + column information directly from database
    """
    return langchain_db.get_table_info()

def run_query(sql: str):
    try:
        return pd.read_sql(sql, engine)
    except Exception as e:
        return str(e)
