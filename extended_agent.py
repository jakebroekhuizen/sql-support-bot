import asyncio
import json
import warnings
from functools import partial

warnings.filterwarnings("ignore", category=Warning)

from dotenv import load_dotenv
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessageGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

import prompts

load_dotenv()

# Load the database
db = SQLDatabase.from_uri("sqlite:///chinook.db")

# ANSI escape codes
BLUE = "\033[94m"
RESET = "\033[0m"

# Load the model
model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4-turbo-preview")


# ------------------------------------------------------------------------------
# Customer Agent
# ------------------------------------------------------------------------------


# Customer info tool
@tool
def get_customer_info(
    *, customer_id: int = None, user_role: str = None, user_id: int = None
):
    """
    Look up customer info given their ID.

    Args:
        customer_id (int): The ID of the target customer to view.
        user_role (str): The role of the user ("employee" or "customer").
        user_id (int): The authenticated user ID, used for RBAC checks.
    """
    # print(
    #     "get_customer_info called with:",
    #     customer_id,
    #     user_role,
    #     user_id,
    # )  # Debug

    # Implement RBAC
    if user_role == "employee":
        return db.run(f"SELECT * FROM customers WHERE CustomerID = {customer_id};")
    elif user_role == "customer" and not customer_id:
        return db.run(f"SELECT * FROM customers WHERE CustomerID = {user_id};")
    elif user_role == "customer" and customer_id != user_id:
        return "Access denied: You can only view your own customer information."

    # Default case or if additional_kwargs is missing
    return "Unable to verify permissions. Please authenticate first."


@tool
def update_customer_info(
    *,
    customer_id: int = None,
    field: str = None,
    new_value: str = None,
    user_role: str = None,
    user_id: int = None,
):
    """
    Update a customer information field to a new value.

    Args:
        customer_id (int): The target customer ID to update.
        field (str): The name of the field to update (e.g., "Address", "Email").
        new_value (str): The new value to assign to that field.
        user_role (str): The role of the authenticated user ("employee" or "customer").
        user_id (int): The authenticated user's ID.

    """
    print(
        "update_customer_info called with:",
        customer_id,
        field,
        new_value,
        user_role,
        user_id,
    )  # Debug

    # For employees, allow update on any record.
    if user_role == "employee":
        db.run(
            f"UPDATE customers SET {field} = '{new_value}' WHERE CustomerID = {customer_id};"
        )
        return f"Customer {customer_id}'s {field} has been updated to: {new_value}"

    # For customers, only allow updating their own record.
    elif user_role == "customer" and not customer_id:
        db.run(
            f"UPDATE customers SET {field} = '{new_value}' WHERE CustomerID = {user_id};"
        )
        return f"Your {field} has been updated to: {new_value}"
    elif user_role == "customer" and customer_id != user_id:
        return "Access denied: You can only update your own customer information."

    return "Unable to verify permissions. Please authenticate first."


def get_customer_info_messages(messages):
    return [SystemMessage(content=prompts.customer_info_prompt)] + messages


def get_update_customer_messages(messages):
    return [SystemMessage(content=prompts.customer_update_prompt)] + messages


update_customer_chain = get_update_customer_messages | model.bind_tools(
    [update_customer_info]
)


# Bind the tools to the model
customer_info_chain = get_customer_info_messages | model.bind_tools([get_customer_info])
customer_update_chain = get_update_customer_messages | model.bind_tools(
    [update_customer_info]
)

# ------------------------------------------------------------------------------
# Music Agent
# ------------------------------------------------------------------------------

# Build the indexes

artists = db._execute("select * from artists")
songs = db._execute("select * from tracks")
artist_retriever = SKLearnVectorStore.from_texts(
    [a["Name"] for a in artists], OpenAIEmbeddings(), metadatas=artists
).as_retriever()
song_retriever = SKLearnVectorStore.from_texts(
    [a["Name"] for a in songs], OpenAIEmbeddings(), metadatas=songs
).as_retriever()


# Get albums by artist tool
@tool
def get_albums_by_artist(artist):
    """Get albums by an artist (or similar artists)."""
    docs = artist_retriever.invoke(artist)
    artist_ids = ", ".join([str(d.metadata["ArtistId"]) for d in docs])
    return db.run(
        f"SELECT Title, Name FROM albums LEFT JOIN artists ON albums.ArtistId = artists.ArtistId WHERE albums.ArtistId in ({artist_ids});",
        include_columns=True,
    )


# Get tracks by artist tool
@tool
def get_tracks_by_artist(artist):
    """Get songs by an artist (or similar artists)."""
    docs = artist_retriever.invoke(artist)
    artist_ids = ", ".join([str(d.metadata["ArtistId"]) for d in docs])
    return db.run(
        f"SELECT tracks.Name as SongName, artists.Name as ArtistName FROM albums LEFT JOIN artists ON albums.ArtistId = artists.ArtistId LEFT JOIN tracks ON tracks.AlbumId = albums.AlbumId WHERE albums.ArtistId in ({artist_ids});",
        include_columns=True,
    )


# Check for songs tool
@tool
def check_for_songs(song_title):
    """Check if a song exists by its name."""
    return song_retriever.invoke(song_title)


song_system_message = """Your job is to help a customer find any songs they are looking for. 

You only have certain tools you can use. If a customer asks you to look something up that you don't know how, politely tell them what you can help with.

When looking up artists and songs, sometimes the artist/song will not be found. In that case, the tools will return information \
on simliar songs and artists. This is intentional, it is not the tool messing up."""


def get_song_messages(messages):
    return [SystemMessage(content=song_system_message)] + messages


song_recc_chain = get_song_messages | model.bind_tools(
    [get_albums_by_artist, get_tracks_by_artist, check_for_songs]
)

# ------------------------------------------------------------------------------
# Generic Agent
# ------------------------------------------------------------------------------


class Router(BaseModel):
    """Call this if you are able to route the user to the appropriate representative."""

    choice: str = Field(
        description="should be one of: music, customer_info, customer_update"
    )


def get_messages(messages):
    return [SystemMessage(content=prompts.router_prompt)] + messages


chain = get_messages | model.bind_tools([Router])


# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------


def add_name(message, name):
    _dict = message.model_dump()
    _dict["name"] = name
    return AIMessage(**_dict)


def _get_last_ai_message(messages):
    for m in messages[::-1]:
        if isinstance(m, AIMessage):
            return m
    return None


def get_user_role_and_Id(db, first_name: str, last_name: str) -> dict:
    """Determine if a user is a customer, employee or neither given their first and last name.
    Returns a dictionary with 'role' and 'id' if found, None otherwise."""
    # Check employees
    employees = db.run(
        f"""
        SELECT EmployeeId FROM employees 
        WHERE FirstName='{first_name}' AND LastName='{last_name}'
    """
    )
    if employees and employees.strip() != "[]":
        try:
            # Remove brackets, parentheses and extract the number
            employee_id = int(employees.strip("[]() ").split(",")[0])
            return {"role": "employee", "id": employee_id}
        except (ValueError, IndexError):
            pass

    # Check customers
    customers = db.run(
        f"""
        SELECT CustomerId FROM customers
        WHERE FirstName='{first_name}' AND LastName='{last_name}'
    """
    )
    if customers and customers.strip() != "[]":
        try:
            # Remove brackets, parentheses and extract the number
            customer_id = int(customers.strip("[]() ").split(",")[0])
            return {"role": "customer", "id": customer_id}
        except (ValueError, IndexError):
            pass

    return None


def _is_tool_call(msg):
    return hasattr(msg, "additional_kwargs") and "tool_calls" in msg.additional_kwargs


def _route(messages):
    last_message = messages[-1]
    if isinstance(last_message, AIMessage):
        if not _is_tool_call(last_message):
            return END
        else:
            if last_message.name == "general":
                tool_calls = last_message.additional_kwargs["tool_calls"]
                if len(tool_calls) > 1:
                    raise ValueError
                tool_call = tool_calls[0]
                return json.loads(tool_call["function"]["arguments"])["choice"]
            else:
                return "tools"
    last_m = _get_last_ai_message(messages)
    if last_m is None:
        return "general"
    if last_m.name == "music":
        return "music"
    elif last_m.name == "customer_info":
        return "customer_info"
    elif last_m.name == "customer_update":
        return "customer_update"
    else:
        return "general"


def _filter_out_routes(messages):
    ms = []
    for m in messages:
        if _is_tool_call(m):
            if m.name == "general":
                continue
        ms.append(m)
    return ms


async def call_tool(messages):
    if messages:
        last_message = messages[-1]
        if not last_message.content and "tool_calls" in last_message.additional_kwargs:
            new_data = last_message.model_dump()

            # Grab the first tool call from additional_kwargs
            tool_call = last_message.additional_kwargs["tool_calls"][0]["function"]
            # Parse the 'arguments' JSON string into an object (if present)
            parsed_args = json.loads(tool_call.get("arguments", "{}"))

            # Find the most recent human message to get user context
            human_message = next(
                (msg for msg in reversed(messages) if isinstance(msg, HumanMessage)),
                None,
            )

            for tc in new_data.get("tool_calls", []):
                if (
                    tc["name"] in ["get_customer_info", "update_customer_info"]
                    and human_message
                ):  # Update get_customer_info tool with required args
                    tc["args"].update(
                        {
                            "user_role": human_message.additional_kwargs.get("role"),
                            "user_id": human_message.additional_kwargs.get("id"),
                        }
                    )
            messages[-1] = AIMessage(**new_data)

    # Use the asynchronous invocation method (ainvoke) instead of invoke.
    tool_messages = await tool_node.ainvoke(messages)
    return tool_messages


# ------------------------------------------------------------------------------
# Build Nodes
# ------------------------------------------------------------------------------

tools = [
    get_albums_by_artist,
    get_tracks_by_artist,
    check_for_songs,
    get_customer_info,
    update_customer_info,
]
tool_node = ToolNode(tools)
general_node = _filter_out_routes | chain | partial(add_name, name="general")
music_node = _filter_out_routes | song_recc_chain | partial(add_name, name="music")
customer_info_node = (
    _filter_out_routes | customer_info_chain | partial(add_name, name="customer_info")
)
customer_update_node = (
    _filter_out_routes
    | customer_update_chain
    | partial(add_name, name="customer_update")
)

# ------------------------------------------------------------------------------
# Define the graph
# ------------------------------------------------------------------------------

memory = SqliteSaver.from_conn_string(":memory:")
graph = MessageGraph()
nodes = {
    "general": "general",
    "music": "music",
    "tools": "tools",
    "customer_info": "customer_info",
    "customer_update": "customer_update",
    END: END,
}
# Define a new graph
workflow = MessageGraph()
workflow.add_node("general", general_node)
workflow.add_node("music", music_node)
workflow.add_node("customer_info", customer_info_node)
workflow.add_node("customer_update", customer_update_node)
workflow.add_node("tools", call_tool)
workflow.add_conditional_edges("general", _route, nodes)
workflow.add_conditional_edges("tools", _route, nodes)
workflow.add_conditional_edges("music", _route, nodes)
workflow.add_conditional_edges("customer_info", _route, nodes)
workflow.add_conditional_edges("customer_update", _route, nodes)
workflow.set_conditional_entry_point(_route, nodes)
graph = workflow.compile()

history = []


async def main():

    print(f"{BLUE}Nellie:{RESET} Hi! I'm Nellie, your intelligent assistant.")
    print(
        "To start, let's get you authenticated. Please enter your first and last name below.\n"
    )

    first_name = input("First name: ")
    last_name = input("Last name: ")
    print("")

    user_info = get_user_role_and_Id(db, first_name, last_name)
    if user_info is None:
        print(
            f"{BLUE}Nellie:{RESET} I'm sorry, I couldn't find you in the system. Please check your spelling and try again."
        )
        return

    print(f"{BLUE}Nellie:{RESET} Welcome, {first_name}!")

    while True:
        user = input(f"{first_name} (q/Q to quit): ")
        if user in {"q", "Q"}:
            print(f"{BLUE}Nellie:{RESET} See you next time!")
            break

        # Store users message along with their role for RBAC
        history.append(
            HumanMessage(
                content=user,
                additional_kwargs={
                    "role": user_info["role"],
                    "id": user_info["id"],
                    "first_name": first_name,
                    "last_name": last_name,
                },
            )
        )
        last_content = None

        async for output in graph.astream(history):
            if END in output or START in output:
                continue
            # Only store the content, don't print intermediate outputs
            for key, value in output.items():
                if isinstance(value, AIMessage):
                    last_content = value.content

        # Print the final output with the "Nellie" label in color
        if last_content:
            print(f"{BLUE}Nellie:{RESET} {last_content}")
            history.append(AIMessage(content=last_content))
        else:
            print(f"{BLUE}Nellie:{RESET} I'm sorry, I couldn't process that request.")


if __name__ == "__main__":
    asyncio.run(main())
