import asyncio
import json
import sys
import time
import warnings
from functools import partial
from typing import Literal, Optional

warnings.filterwarnings("ignore", category=Warning)

import ast

from dotenv import load_dotenv
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, FunctionNode, MessageGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, RunContext, interrupt
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


def debug_print(msg):
    print(f"DEBUG: {msg}", file=sys.stderr)


# ------------------------------------------------------------------------------
# Customer Agent
# ------------------------------------------------------------------------------


# Customer info tool
@tool
def get_customer_info(
    *, customer_id: Optional[int] = None, user_role: str = None, user_id: int = None
):
    """
    Look up customer info given their ID.

    Args:
        customer_id (Optional[int]): The ID of the target customer to view.
        user_role (str): The role of the user ("employee" or "customer").
        user_id (int): The authenticated user ID, used for RBAC checks.
    """
    try:
        if user_role is None:
            return "Error: User role is required for authentication."

        if user_id is None:
            return "Error: User ID is required for authentication."

        # Implement RBAC
        if user_role == "employee":
            return db.run(f"SELECT * FROM customers WHERE CustomerID = {customer_id};")
        elif user_role == "customer" and not customer_id:
            return db.run(f"SELECT * FROM customers WHERE CustomerID = {user_id};")
        elif user_role == "customer" and customer_id != user_id:
            return "Access denied: You can only view your own customer information."

        # Default case or if additional_kwargs is missing
        return "Unable to verify permissions. Please authenticate first."
    except Exception as e:
        return f"Unexpected error in get_customer_info: {e}"


@tool
def update_customer_info(
    *,
    customer_id: Optional[int] = None,
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
    try:
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
    except Exception as e:
        return f"Unexpected error in update_customer_info: {e}"


def get_customer_messages(messages):
    # Find the human message to get the role
    try:
        if messages:
            human_message = next(
                (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
            )
            if human_message:
                user_role = human_message.additional_kwargs.get("role")
                if user_role == "employee":
                    return [
                        SystemMessage(content=prompts.customer_employee_prompt)
                    ] + messages
                elif user_role == "customer":
                    return [
                        SystemMessage(content=prompts.customer_customer_prompt)
                    ] + messages

        # Default case if no human message or role found
        return [SystemMessage(content=prompts.customer_customer_prompt)] + messages
    except Exception as e:
        return f"Unexpected error in get_customer_messages: {e}"


# Bind both tools
customer_chain = get_customer_messages | model.bind_tools(
    [get_customer_info, update_customer_info]
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
    try:
        docs = artist_retriever.invoke(artist)
        artist_ids = ", ".join([str(d.metadata["ArtistId"]) for d in docs])
        return db.run(
            f"SELECT Title, Name FROM albums LEFT JOIN artists ON albums.ArtistId = artists.ArtistId WHERE albums.ArtistId in ({artist_ids});",
            include_columns=True,
        )
    except Exception as e:
        return f"Unexpected error in get_albums_by_artist: {e}"


# Get tracks by artist tool
@tool
def get_tracks_by_artist(artist):
    """Get songs by an artist (or similar artists)."""
    try:
        docs = artist_retriever.invoke(artist)
        artist_ids = ", ".join([str(d.metadata["ArtistId"]) for d in docs])
        return db.run(
            f"SELECT tracks.Name as SongName, artists.Name as ArtistName FROM albums LEFT JOIN artists ON albums.ArtistId = artists.ArtistId LEFT JOIN tracks ON tracks.AlbumId = albums.AlbumId WHERE albums.ArtistId in ({artist_ids});",
            include_columns=True,
        )
    except Exception as e:
        return f"Unexpected error in get_tracks_by_artist: {e}"


# Check for songs tool
@tool
def check_for_songs(song_title):
    """Check if a song exists by its name."""
    return song_retriever.invoke(song_title)


def get_song_messages(messages):
    return [SystemMessage(content=prompts.song_prompt)] + messages


song_recc_chain = get_song_messages | model.bind_tools(
    [get_albums_by_artist, get_tracks_by_artist, check_for_songs]
)

# ------------------------------------------------------------------------------
# Invoice Agent
# ------------------------------------------------------------------------------


@tool
def list_invoices_for_customer(
    *, customer_id: Optional[int] = None, user_role: str = None, user_id: int = None
):
    """
    List all invoices for the given customer.

    Employees can list any customer's invoices.
    Customers can only list their own.
    """
    try:
        if user_role == "employee":
            query = f"SELECT * FROM invoices WHERE CustomerId = {customer_id};"
        elif user_role == "customer":
            if customer_id is None or customer_id == user_id:
                query = f"SELECT * FROM invoices WHERE CustomerId = {user_id};"
            else:
                return "Access denied: you can only view your own invoices."
        else:
            return "Access denied or unable to verify permissions."

        return db.run(query)
    except Exception as e:
        return f"Unexpected error in list_invoices_for_customer: {e}"


@tool
def get_invoice_details(*, invoice_id: int, user_role: str = None, user_id: int = None):
    """
    Show line items (invoice_items) for a given invoice ID,
    if the user has permissions to view it.
    """
    try:
        # Check if user is employee or the customer who owns this invoice
        invoice_info = db.run(
            f"SELECT CustomerId FROM invoices WHERE InvoiceId = {invoice_id};"
        )

        if not invoice_info or invoice_info == "[]":
            return f"Invoice {invoice_id} not found."

        try:
            info_str = invoice_info.strip("[]() ")
            info_str = info_str.replace(",", "")  # remove any trailing commas
            inv_customer_id = int(info_str)
        except:
            return f"Error parsing invoice data: {invoice_info}"

        if user_role == "employee" or (
            user_role == "customer" and inv_customer_id == user_id
        ):
            # Return invoice items
            query = f"""
                SELECT invoice_items.*, tracks.Name as TrackName
                FROM invoice_items
                JOIN tracks ON invoice_items.TrackId = tracks.TrackId
                WHERE InvoiceId = {invoice_id};
            """
            return db.run(query)
        else:
            return "Access denied: this invoice does not belong to you."
    except Exception as e:
        return f"Unexpected error in get_invoice_details: {e}"


def get_invoice_messages(messages):
    # Find the human message to get the role
    try:
        if messages:
            human_message = next(
                (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
            )
            if human_message:
                user_role = human_message.additional_kwargs.get("role")
                if user_role == "employee":
                    return [
                        SystemMessage(content=prompts.invoice_employee_prompt)
                    ] + messages
                elif user_role == "customer":
                    return [
                        SystemMessage(content=prompts.invoice_customer_prompt)
                    ] + messages

        # Default case if no human message or role found
        return [SystemMessage(content=prompts.invoice_customer_prompt)] + messages
    except Exception as e:
        return f"Unexpected error in get_invoice_messages: {e}"


invoice_chain = get_invoice_messages | model.bind_tools(
    [list_invoices_for_customer, get_invoice_details]
)


# ------------------------------------------------------------------------------
# Refund Agent
# ------------------------------------------------------------------------------


@tool
def issue_full_refund(*, invoice_id: int, user_role: str = None, user_id: int = None):
    """
    Issue a full refund for an invoice.

    Only employees can do this. The refund is processed by setting the invoice total to 0.
    A negative line item is inserted to record the refund.
    """
    try:
        if user_role != "employee":
            return "Access denied: Only employees can issue refunds."

        # Check if invoice exists and get the current total
        invoice_info = db.run(
            f"SELECT Total FROM invoices WHERE InvoiceId = {invoice_id};"
        )
        if not invoice_info or invoice_info == "[]":
            return f"Invoice {invoice_id} not found."

        try:
            current_total = float(invoice_info.strip("[](), "))
        except Exception as e:
            return f"Error parsing invoice data: {invoice_info}. Exception: {e}"

        refund_amount = current_total  # For a full refund

        # Update the invoice total to zero (full refund)
        update_query = f"""
            UPDATE invoices
            SET Total = 0
            WHERE InvoiceId = {invoice_id};
        """
        db.run(update_query)

        # Insert a negative invoice_item to record the refund transaction
        insert_query = f"""
            INSERT INTO invoice_items (InvoiceId, TrackId, UnitPrice, Quantity)
            VALUES ({invoice_id}, 0, {-refund_amount}, 1);
        """
        db.run(insert_query)

        return (
            f"Full refund processed for Invoice {invoice_id}.\n"
            f"Previous Total: {current_total}\n"
            f"Refund Amount: {refund_amount}\n"
            f"New Total: 0"
        )
    except Exception as e:
        return f"Unexpected error in issue_full_refund: {e}"


@tool
def refund_line_item(
    *, invoice_id: int, track_name: str, user_role: str = None, user_id: int = None
):
    """
    Issue a partial refund for a specific track on a given invoice.

    Only employees can do this. The refund is processed by subtracting
    the line item's total price (UnitPrice * Quantity) from the invoice total.
    A negative invoice item is inserted to record the refund transaction.
    """
    try:
        if user_role != "employee":
            return "Access denied: Only employees can issue partial refunds."

        # Confirm the invoice exists & get current total
        invoice_info = db.run(
            f"SELECT Total FROM invoices WHERE InvoiceId = {invoice_id};"
        )
        if not invoice_info or invoice_info == "[]":
            return f"Invoice {invoice_id} not found."

        try:
            current_total = ast.literal_eval(invoice_info)[0][0]
        except Exception as e:
            return f"Error parsing invoice total: {invoice_info}. Exception: {e}"

        # Look up the line item for the given track on this invoice
        # Returns the line item (if any) for track name.
        line_item_str = db.run(
            f"""
            SELECT invoice_items.InvoiceLineId,
                invoice_items.UnitPrice,
                invoice_items.Quantity,
                tracks.TrackId
            FROM invoice_items
            JOIN tracks ON invoice_items.TrackId = tracks.TrackId
            WHERE invoice_items.InvoiceId = {invoice_id}
            AND tracks.Name = '{track_name}';
        """
        )
        if not line_item_str or line_item_str == "[]":
            return f"Track '{track_name}' not found in Invoice {invoice_id}."

        # Parse the line item data
        try:
            line_items = ast.literal_eval(line_item_str)
        except Exception as e:
            return f"Error parsing line item data: {line_item_str}. Exception: {e}"

        # For simplicity, assume there's only one matching line item
        line_item_id, unit_price, quantity, track_id = line_items[0]

        # Calculate the refund amount = UnitPrice * Quantity
        refund_amount = unit_price * quantity
        new_total = current_total - refund_amount
        if new_total < 0:
            return (
                f"Refund amount ({refund_amount}) exceeds current invoice total "
                f"({current_total}). Cannot proceed."
            )

        # Update the invoice total
        update_query = f"""
            UPDATE invoices
            SET Total = {new_total}
            WHERE InvoiceId = {invoice_id};
        """
        db.run(update_query)

        # Insert a negative invoice_item row to record the refund for that track
        insert_query = f"""
            INSERT INTO invoice_items (InvoiceId, TrackId, UnitPrice, Quantity)
            VALUES ({invoice_id}, {track_id}, {-refund_amount}, 1);
        """
        db.run(insert_query)

        return (
            f"Refund for track '{track_name}' (Invoice {invoice_id}) processed.\n"
            f"Track Price: {unit_price} x {quantity} = {refund_amount}\n"
            f"Previous Total: {current_total}\n"
            f"New Total: {new_total}"
        )
    except Exception as e:
        return f"Unexpected error in refund_line_item: {e}"


def get_refund_messages(messages):
    debug_print("get_refund_messages called")
    # Find the human message to get the role
    try:
        if messages:
            human_message = next(
                (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
            )
            debug_print(f"Found human message: {human_message is not None}")
            if human_message:
                user_role = human_message.additional_kwargs.get("role")
                debug_print(f"User role: {user_role}")
                if user_role == "employee":
                    debug_print("Using employee prompt")
                    return [
                        SystemMessage(content=prompts.refund_employee_prompt)
                    ] + messages
                elif user_role == "customer":
                    debug_print("Using customer prompt")
                    return [
                        SystemMessage(content=prompts.refund_customer_prompt)
                    ] + messages

        # Default case if no human message or role found
        debug_print("Using default customer prompt")
        return [SystemMessage(content=prompts.refund_customer_prompt)] + messages
    except Exception as e:
        debug_print(f"Error in get_refund_messages: {str(e)}")
        return f"Unexpected error in get_refund_messages: {e}"


refund_chain = get_refund_messages | model.bind_tools(
    [issue_full_refund, refund_line_item]
)


# Modify the refund_chain to include human approval
def refund_with_approval(messages):
    """
    Process refund requests with human-in-the-loop approval.
    """
    try:
        # First, run the normal refund chain to generate the tool call
        result = refund_chain.invoke(messages)

        # Check if the result contains a refund tool call
        if isinstance(result, AIMessage) and "tool_calls" in result.additional_kwargs:
            tool_calls = result.additional_kwargs["tool_calls"]
            for tc in tool_calls:
                if tc["function"]["name"] in ["issue_full_refund", "refund_line_item"]:
                    # Extract refund details
                    refund_details = tc["function"]["arguments"]

                    # Request human approval
                    approval_response = interrupt(
                        {
                            "question": "A refund request has been initiated. Do you approve processing the refund?",
                            "refund_details": refund_details,
                            "options": ["Approve", "Deny"],
                        }
                    )

                    # Process based on approval
                    if (
                        approval_response
                        and approval_response.get("choice") == "Approve"
                    ):
                        # Continue with the original tool call
                        return result
                    else:
                        # Replace with a denial message
                        return AIMessage(
                            content="The refund request has been reviewed but was not approved by management. Please contact the customer to discuss alternative solutions.",
                            name="refund",
                        )

        # If no refund tool call was found, just return the original result
        return result
    except Exception as e:
        return AIMessage(
            content=f"There was an error processing the refund request: {str(e)}. Please try again later.",
            name="refund",
        )


# ------------------------------------------------------------------------------
# Router Agent
# ------------------------------------------------------------------------------


class Router(BaseModel):
    """Call this if you are able to route the user to the appropriate representative."""

    choice: str = Field(
        description="should be one of: music, customer, invoice, refund"
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


def get_latest_human_with_role(messages):
    """
    Scans the message list to find the last human message with a role.
    """
    for m in reversed(messages):
        if isinstance(m, HumanMessage) and m.additional_kwargs.get("role") is not None:
            return m
    return None


def extract_refund_details(messages):
    """
    Scans the message list to find the last refund tool call and extracts its arguments.
    """
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            tool_calls = msg.additional_kwargs.get("tool_calls", [])
            for tc in tool_calls:
                if tc["function"]["name"] in ["issue_full_refund", "refund_line_item"]:
                    return tc["function"]["arguments"]
    return {}


def get_user_role_and_Id(db, first_name: str, last_name: str) -> dict:
    """Determine if a user is a customer, employee or neither given their first and last name.
    Returns a dictionary with 'role' and 'id' if found, None otherwise."""
    try:
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
    except Exception as e:
        return f"Unexpected error in get_user_role_and_Id: {e}"


def _is_tool_call(msg):
    return hasattr(msg, "additional_kwargs") and "tool_calls" in msg.additional_kwargs


def _is_refund_tool_call(message):
    """Check if the message contains a refund tool call."""
    debug_print(f"Checking if message is refund tool call")
    if not _is_tool_call(message):
        debug_print("Not a tool call")
        return False

    tool_calls = message.additional_kwargs.get("tool_calls", [])
    debug_print(f"Found {len(tool_calls)} tool calls")

    for tc in tool_calls:
        debug_print(f"Tool call: {tc}")
        if "function" in tc and tc["function"]["name"] in [
            "issue_full_refund",
            "refund_line_item",
        ]:
            debug_print("Found refund tool call!")
            return True
    debug_print("No refund tool calls found")
    return False


def _route(messages):
    debug_print(f"_route called with {len(messages)} messages")
    last_message = messages[-1]
    debug_print(f"Last message type: {type(last_message)}")

    if isinstance(last_message, AIMessage):
        debug_print(
            f"Last message is AIMessage with name: {getattr(last_message, 'name', 'None')}"
        )
        debug_print(
            f"Last message has tool_calls: {'tool_calls' in last_message.additional_kwargs}"
        )

        if not _is_tool_call(last_message):
            debug_print("Not a tool call, returning END")
            return END
        else:
            debug_print("Is a tool call")
            if last_message.name == "general":
                tool_calls = last_message.additional_kwargs["tool_calls"]
                debug_print(f"General message with {len(tool_calls)} tool calls")
                if len(tool_calls) > 1:
                    raise ValueError
                tool_call = tool_calls[0]
                choice = json.loads(tool_call["function"]["arguments"])["choice"]
                debug_print(f"Routing to choice: {choice}")
                return choice
            # Check if this is a refund tool call
            elif _is_refund_tool_call(last_message):
                debug_print("Detected refund tool call, routing to refund_approval")
                return "refund_approval"  # Route to approval first
            else:
                debug_print("Other tool call, routing to tools")
                return "tools"

    debug_print("Checking last AI message")
    last_m = _get_last_ai_message(messages)
    if last_m is None:
        debug_print("No last AI message, routing to general")
        return "general"

    debug_print(f"Last AI message name: {getattr(last_m, 'name', 'None')}")
    if last_m.name == "music":
        return "music"
    elif last_m.name == "customer":
        return "customer"
    elif last_m.name == "invoice":
        return "invoice"
    elif last_m.name == "refund":
        return "refund"
    else:
        debug_print("Default routing to general")
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
    try:
        if messages:
            last_message = messages[-1]
            if (
                not last_message.content
                and "tool_calls" in last_message.additional_kwargs
            ):
                new_data = last_message.model_dump()

                # Find the most recent human message to get user context
                human_message = get_latest_human_with_role(messages)

                for tc in new_data.get("tool_calls", []):
                    if (
                        tc["name"]
                        in [
                            "get_customer_info",
                            "update_customer_info",
                            "list_invoices_for_customer",
                            "get_invoice_details",
                            "issue_full_refund",
                            "refund_line_item",
                        ]
                        and human_message
                    ):  # Update tool with required args
                        tc["args"].update(
                            {
                                "user_role": human_message.additional_kwargs.get(
                                    "role"
                                ),
                                "user_id": human_message.additional_kwargs.get("id"),
                            }
                        )
                messages[-1] = AIMessage(**new_data)

        # asychronously invoke
        tool_messages = await tool_node.ainvoke(messages)
        return tool_messages
    except Exception as e:
        return f"Unexpected error in call_tool: {e}"


# ------------------------------------------------------------------------------
# Build Nodes
# ------------------------------------------------------------------------------


@functional_node(include_context=True)
def refund_approval_node(
    messages, context: RunContext
) -> Command[Literal["refund", "general"]]:
    # 1) Extract details from the conversation
    refund_details = extract_refund_details(messages)

    # 2) Use interrupt(..., context=context)
    approval = interrupt(
        {
            "question": f"A refund request has been initiated with details: {refund_details}. Do you approve?",
            "options": ["Approve", "Deny"],
        },
        context=context,
    )

    # 3) Resume
    if approval and approval.get("choice") == "Approve":
        return Command(goto="refund")
    else:
        return Command(goto="general")


tools = [
    get_albums_by_artist,
    get_tracks_by_artist,
    check_for_songs,
    get_customer_info,
    update_customer_info,
    list_invoices_for_customer,
    get_invoice_details,
    issue_full_refund,
    refund_line_item,
]
tool_node = ToolNode(tools)
general_node = _filter_out_routes | chain | partial(add_name, name="general")
music_node = _filter_out_routes | song_recc_chain | partial(add_name, name="music")
customer_node = _filter_out_routes | customer_chain | partial(add_name, name="customer")
invoice_node = _filter_out_routes | invoice_chain | partial(add_name, name="invoice")
refund_regular_node = (
    _filter_out_routes | refund_chain | partial(add_name, name="refund")
)

# ------------------------------------------------------------------------------
# Define the graph
# ------------------------------------------------------------------------------

graph = MessageGraph()
nodes = {
    "general": "general",
    "music": "music",
    "tools": "tools",
    "customer": "customer",
    "invoice": "invoice",
    "refund": "refund",
    "refund_approval": "refund_approval",
    END: END,
}
# Define a new graph
workflow = MessageGraph()
workflow.add_node("general", general_node)
workflow.add_node("music", music_node)
workflow.add_node("customer", customer_node)
workflow.add_node("invoice", invoice_node)
workflow.add_node("refund", refund_regular_node)
workflow.add_node("refund_approval", refund_approval_node)
workflow.add_node("tools", call_tool)
workflow.add_conditional_edges("general", _route, nodes)
workflow.add_conditional_edges("tools", _route, nodes)
workflow.add_conditional_edges("music", _route, nodes)
workflow.add_conditional_edges("customer", _route, nodes)
workflow.add_conditional_edges("invoice", _route, nodes)
workflow.add_conditional_edges("refund", _route, nodes)
workflow.add_conditional_edges("refund_approval", _route, nodes)
workflow.set_conditional_entry_point(_route, nodes)

history = []


async def main():
    async with AsyncSqliteSaver.from_conn_string(":memory:") as memory:
        graph = workflow.compile(checkpointer=memory)

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

        # Generate a unique thread ID for this conversation
        thread_id = f"thread_{first_name}_{last_name}_{int(time.time())}"
        thread_config = {"configurable": {"thread_id": thread_id}}

        while True:
            user = input(f"{first_name} (q/Q to quit): ")
            if user in {"q", "Q"}:
                print(f"{BLUE}Nellie:{RESET} See you next time!")
                break

            # Store user's message along with their role for RBAC
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

            async for output in graph.astream(history, thread_config):
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
                print(
                    f"{BLUE}Nellie:{RESET} I'm sorry, I couldn't process that request."
                )


if __name__ == "__main__":
    asyncio.run(main())
