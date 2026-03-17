from typing import TypedDict, Optional, Dict, Any,List

class AgentState(TypedDict):
    question: str
    route: Optional[str]   # "sql" or "rag"
    response: Optional[Dict[str, Any]]
    history : List[Dict[str, str]]
    chosen_agent : str
    session_id : str

