# from pydantic import BaseModel, Field
from typing_extensions import TypedDict, List
from langgraph.graph.message import add_messages
from typing import Annotated

class State(TypedDict) :
    """
    Represent the structure of the state of the graph
    """
    messages: Annotated[List, add_messages]
    