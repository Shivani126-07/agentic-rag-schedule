 import os
from datetime import datetime

import chromadb
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
from google.genai import types


# =========================================================
# CONFIGURATION
# =========================================================

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is missing.")

client = genai.Client(api_key=API_KEY)

app = FastAPI(
    title="Agentic RAG Schedule Assistant",
    description="AI assistant for managing a 30-day schedule."
)


# =========================================================
# CHROMADB
# =========================================================

chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(
    name="schedule"
)


# =========================================================
# SAMPLE SCHEDULE - NEXT 30 DAYS
# =========================================================

schedule = [
    {
        "id": "event_1",
        "date": "2026-08-17",
        "time": "10:00 AM",
        "title": "Team Meeting",
        "type": "meeting",
        "description": "Discuss project progress and assign tasks."
    },
    {
        "id": "event_2",
        "date": "2026-08-18",
        "time": "02:00 PM",
        "title": "AI Workshop",
        "type": "workshop",
        "description": "Learn about generative AI and AI agents."
    },
    {
        "id": "event_3",
        "date": "2026-08-20",
        "time": "11:00 AM",
        "title": "Doctor Appointment",
        "type": "appointment",
        "description": "Regular health appointment."
    },
    {
        "id": "event_4",
        "date": "2026-08-21",
        "time": "03:00 PM",
        "title": "Project Review",
        "type": "meeting",
        "description": "Review the current project."
    },
    {
        "id": "event_5",
        "date": "2026-08-22",
        "time": "09:00 AM",
        "title": "Complete Assignment",
        "type": "task",
        "description": "Complete and submit the AI assignment."
    },
    {
        "id": "event_6",
        "date": "2026-08-25",
        "time": "01:00 PM",
        "title": "Client Meeting",
        "type": "meeting",
        "description": "Discuss project requirements with the client."
    },
    {
        "id": "event_7",
        "date": "2026-08-28",
        "time": "10:00 AM",
        "title": "Python Workshop",
        "type": "workshop",
        "description": "Advanced Python programming workshop."
    },
    {
        "id": "event_8",
        "date": "2026-09-01",
        "time": "04:00 PM",
        "title": "Team Planning",
        "type": "meeting",
        "description": "Plan tasks for the upcoming project."
    }
]


# =========================================================
# HELPER - CREATE DOCUMENT
# =========================================================

def make_document(event):

    return (
        f"Date: {event['date']}\n"
        f"Time: {event['time']}\n"
        f"Title: {event['title']}\n"
        f"Type: {event['type']}\n"
        f"Description: {event['description']}"
    )


# =========================================================
# LOAD SCHEDULE INTO CHROMADB
# =========================================================

def load_schedule():

    # Avoid duplicate loading
    if collection.count() > 0:
        return

    documents = []
    ids = []
    metadatas = []

    for event in schedule:

        document = make_document(event)

        embedding_result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=document
        )

        embedding = embedding_result.embeddings[0].values

        documents.append(document)
        ids.append(event["id"])

        metadatas.append({
            "id": event["id"],
            "date": event["date"],
            "time": event["time"],
            "title": event["title"],
            "type": event["type"]
        })

        collection.add(
            ids=[event["id"]],
            documents=[document],
            embeddings=[embedding],
            metadatas=[metadatas[-1]]
        )


load_schedule()


# =========================================================
# TOOL 1 - GET SCHEDULE
# =========================================================

def get_schedule(query):

    embedding_result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query
    )

    query_embedding = embedding_result.embeddings[0].values

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    if not results["documents"] or not results["documents"][0]:
        return "No matching schedule found."

    output = []

    for i, document in enumerate(results["documents"][0]):

        metadata = results["metadatas"][0][i]

        output.append(
            f"Event ID: {metadata.get('id')}\n"
            f"{document}"
        )

    return "\n\n".join(output)


# =========================================================
# TOOL 2 - UPDATE SCHEDULE
# =========================================================

def update_schedule(
    action,
    event_id=None,
    date_value=None,
    time_value=None,
    title=None,
    event_type=None,
    description=None
):

    # -------------------------
    # ADD
    # -------------------------

    if action == "add":

        new_id = f"event_{len(schedule) + 1}"

        event = {
            "id": new_id,
            "date": date_value,
            "time": time_value,
            "title": title or "New Event",
            "type": event_type or "task",
            "description": description or ""
        }

        schedule.append(event)

        document = make_document(event)

        embedding_result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=document
        )

        collection.add(
            ids=[new_id],
            documents=[document],
            embeddings=[embedding_result.embeddings[0].values],
            metadatas=[{
                "id": new_id,
                "date": event["date"],
                "time": event["time"],
                "title": event["title"],
                "type": event["type"]
            }]
        )

        return f"Added event '{event['title']}' on {event['date']} at {event['time']}."


    # -------------------------
    # UPDATE
    # -------------------------

    if action == "update":

        if not event_id:
            return "Event ID is required for update."

        event = next(
            (e for e in schedule if e["id"] == event_id),
            None
        )

        if not event:
            return f"Event {event_id} not found."

        if date_value:
            event["date"] = date_value

        if time_value:
            event["time"] = time_value

        if title:
            event["title"] = title

        if event_type:
            event["type"] = event_type

        if description:
            event["description"] = description

        document = make_document(event)

        embedding_result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=document
        )

        collection.update(
            ids=[event_id],
            documents=[document],
            embeddings=[embedding_result.embeddings[0].values],
            metadatas=[{
                "id": event["id"],
                "date": event["date"],
                "time": event["time"],
                "title": event["title"],
                "type": event["type"]
            }]
        )

        return f"Updated event '{event['title']}'. New time: {event['time']}."


    # -------------------------
    # DELETE
    # -------------------------

    if action == "delete":

        if not event_id:
            return "Event ID is required for deletion."

        event = next(
            (e for e in schedule if e["id"] == event_id),
            None
        )

        if not event:
            return f"Event {event_id} not found."

        schedule.remove(event)

        collection.delete(
            ids=[event_id]
        )

        return f"Deleted event '{event['title']}'."


    return "Invalid action."


# =========================================================
# GEMINI TOOLS
# =========================================================

tools = [
    {
        "function_declarations": [

            {
                "name": "get_schedule",
                "description": "Retrieve relevant schedule information based on date, time, event type, or user question.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "query": {
                            "type": "STRING",
                            "description": "The user's schedule question."
                        }
                    },
                    "required": ["query"]
                }
            },

            {
                "name": "update_schedule",
                "description": "Add, update, or delete schedule entries.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {

                        "action": {
                            "type": "STRING",
                            "enum": ["add", "update", "delete"]
                        },

                        "event_id": {
                            "type": "STRING"
                        },

                        "date_value": {
                            "type": "STRING"
                        },

                        "time_value": {
                            "type": "STRING"
                        },

                        "title": {
                            "type": "STRING"
                        },

                        "event_type": {
                            "type": "STRING"
                        },

                        "description": {
                            "type": "STRING"
                        }
                    },
                    "required": ["action"]
                }
            }
        ]
    }
]


# =========================================================
# AGENT
# =========================================================

def run_agent(user_query):

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=user_query
                )
            ]
        )
    ]

    for _ in range(5):

        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=contents,
            config={
                "tools": tools,
                "system_instruction": """
You are an Agentic RAG Schedule Assistant.

You manage the user's schedule for the next 30 days.

Use get_schedule when the user asks about existing events,
meetings, workshops, tasks, appointments, or free time.

Use update_schedule when the user wants to add, update,
move, or delete an event.

For update or delete:
First use get_schedule to find the correct Event ID,
then use update_schedule.

Never invent schedule information.
Keep answers short and clear.
"""
            }
        )

        if not response.function_calls:
            return response.text

        contents.append(
            response.candidates[0].content
        )

        for function_call in response.function_calls:

            if function_call.name == "get_schedule":

                result = get_schedule(
                    function_call.args["query"]
                )

            elif function_call.name == "update_schedule":

                result = update_schedule(
                    action=function_call.args.get("action"),
                    event_id=function_call.args.get("event_id"),
                    date_value=function_call.args.get("date_value"),
                    time_value=function_call.args.get("time_value"),
                    title=function_call.args.get("title"),
                    event_type=function_call.args.get("event_type"),
                    description=function_call.args.get("description")
                )

            else:

                result = "Unknown tool."

            function_response = types.Part.from_function_response(
                name=function_call.name,
                response={
                    "result": result
                }
            )

            contents.append(
                types.Content(
                    role="user",
                    parts=[function_response]
                )
            )

    return "I could not complete the request."


# =========================================================
# FASTAPI
# =========================================================

class ChatRequest(BaseModel):

    message: str


@app.get("/")
def home():

    return {
        "message": "Agentic RAG Schedule Assistant is running!"
    }


@app.post("/chat")
def chat(request: ChatRequest):

    answer = run_agent(request.message)

    return {
        "answer": answer
    }
