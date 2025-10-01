import time
import logging
import os
import re
import boto3

from typing import Optional

from pymongo import MongoClient
from langchain_aws.embeddings import BedrockEmbeddings
from botocore.exceptions import ClientError
from strands import Agent, tool
from strands.models import BedrockModel
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.memory import MemoryClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = BedrockAgentCoreApp()

# Memory configuration for Bedrock AgentCore Memory service
MEMORY_ID = os.getenv("MEMORY_ID")
MEMORY_NAMESPACE_TEMPLATE = os.getenv("MEMORY_NAMESPACE", "/travel/{sessionId}")
MEMORY_TOP_K = int(os.getenv("MEMORY_TOP_K", "5"))
MEMORY_PROMPT_PREFIX = os.getenv(
    "MEMORY_PROMPT_PREFIX",
    "Here are details from our recent conversation that may help:\n",
)


def _is_valid_memory_id(value: Optional[str]) -> bool:
    """Return True when the provided memory identifier matches AgentCore format."""
    if not value:
        return False
    pattern = re.compile(r"[a-zA-Z][a-zA-Z0-9-_]{0,99}-[a-zA-Z0-9]{10}$")
    return bool(pattern.fullmatch(value))


if MEMORY_ID and not _is_valid_memory_id(MEMORY_ID):
    logger.warning(
        "Provided MEMORY_ID (%s) does not match the expected format. Short-term memory is disabled until a valid ID is configured.",
        MEMORY_ID,
    )
    MEMORY_ID = None

memory_client: Optional[MemoryClient] = None
if MEMORY_ID:
    try:
        memory_client = MemoryClient()
        logger.info("Initialized Bedrock AgentCore Memory client")
    except Exception as memory_init_error:
        logger.warning(
            "Unable to initialize Bedrock AgentCore Memory client: %s", memory_init_error
        )
        memory_client = None

@tool
def current_time() -> int:
    """Gets the current time in seconds"""
    logger.info("Getting current time in seconds")
    return int(time.time())

@tool
def current_month() -> str:
    """Gets the current month"""
    logger.info("Getting current month")
    return time.strftime("%B")

@tool
def place_lookup_by_country(query_str: str) -> str:
    """Retrieve places by country name
    
    Args:
        query_str: The country name to search for
    
    Returns:
        List of place names in the specified country
    """
    logger.info(f"Looking up places by country: {query_str}")
    client = get_mongo_client()
    # get database and collection
    collection = get_travel_collection(client)
    res = ""
    res = collection.aggregate(
        [
            {"$match": {"Country": {"$regex": query_str, "$options": "i"}}},
            {"$project": {"Place Name": 1}},
        ]
    )
    places = []
    for place in res:
        places.append(place["Place Name"])
    logger.info(f"Found {len(places)} places in country: {query_str}")
    return str(places)

def get_travel_collection(client):
    logger.info("Getting travel collection from MongoDB")
    db = client['Integration']
    collection = db['test_csv_load']
    return collection

@tool
def place_lookup_by_name(query_str: str) -> str:
    """Retrieve place information by place name
    
    Args:
        query_str: The place name to search for
    
    Returns:
        Detailed information about the place
    """
    logger.info(f"Looking up place by name: {query_str}")
    client = get_mongo_client()
    collection = get_travel_collection(client)
    res = ""
    filter = {
        "$or": [
            {"Place Name": {"$regex": query_str, "$options": "i"}},
            {"Country": {"$regex": query_str, "$options": "i"}},
        ]
    }
    project = {"_id": 0}

    res = collection.find_one(filter=filter, projection=project)
    logger.info(f"Found place details for: {query_str}")
    return str(res)

@tool
def place_best_time_lookup(query_str: str) -> str:
    """Retrieve place's best time to visit
    
    Args:
        query_str: The place name to search for
    
    Returns:
        Best time to visit the specified place
    """
    logger.info(f"Looking up best time to visit for place: {query_str}")
    client = get_mongo_client()
    collection = get_travel_collection(client)
    res = ""
    filter = {
        "$or": [
            {"Place Name": {"$regex": query_str, "$options": "i"}},
            {"Country": {"$regex": query_str, "$options": "i"}},
        ]
    }
    project = {"Best Time To Visit": 1, "_id": 0}

    res = collection.find_one(filter=filter, projection=project)
    logger.info(f"Found best time to visit for: {query_str}")
    return str(res)

# Setup bedrock
def setup_bedrock():
    """Initialize the Bedrock runtime."""
    logger.info("Setting up Bedrock runtime client")
    return boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",
    )


@tool
def mongodb_search(query: str) -> str:
    """Retrieve place information by place features using semantic search
    
    Args:
        query: Description of place features to search for
    
    Returns:
        Places and their details matching the specified features
    """
    logger.info(f"Performing semantic search for place features: {query}")
    bedrock_runtime = setup_bedrock()
    embeddings = BedrockEmbeddings(
        client=bedrock_runtime,
        model_id="amazon.titan-embed-text-v1",
    )
    
    client = get_mongo_client()
    collection = get_travel_collection(client)
    
    field_name_to_be_vectorized = "About Place"

    logger.info("Generating embeddings for query")
    text_as_embeddings = embeddings.embed_documents([query])
    embedding_value = text_as_embeddings[0]

    # get the vector search results based on the filter conditions.
    logger.info("Performing vector search in MongoDB")
    response = collection.aggregate(
        [
            {
                "$vectorSearch": {
                    "index": "travel_vector_index",
                    "path": "details_embedding",
                    "queryVector": embedding_value,
                    "numCandidates": 200,
                    "limit": 10,
                }
            },
            {
                "$project": {
                    "score": {"$meta": "vectorSearchScore"},
                    field_name_to_be_vectorized: 1,
                    "_id": 0,
                }
            },
        ]
    )

    # Result is a list of docs with the array fields
    docs = list(response)
    logger.info(f"Found {len(docs)} results from vector search")

    # Extract an array field from the docs
    array_field = [doc[field_name_to_be_vectorized] for doc in docs]

    # Join array elements into a string
    llm_input_text = "\n \n".join(str(elem) for elem in array_field)

    # utility
    newline, bold, unbold = "\n", "\033[1m", "\033[0m"
    logger.info(
        newline
        + bold
        + "Given Input From MongoDB Vector Search: "
        + unbold
        + newline
        + llm_input_text
        + newline
    )

    return llm_input_text


def _format_memory_namespace(session_id: Optional[str], actor_id: Optional[str]) -> Optional[str]:
    """Resolve the namespace pattern used by AgentCore Memory."""
    if not session_id:
        return None

    resolved_actor = actor_id or session_id or "default"
    try:
        return MEMORY_NAMESPACE_TEMPLATE.format(sessionId=session_id, actorId=resolved_actor)
    except Exception as namespace_error:
        logger.warning("Failed to format memory namespace: %s", namespace_error)
        return None


def _retrieve_memory_context(namespace: Optional[str], query: str, session_id: Optional[str], actor_id: Optional[str]) -> list[str]:
    """Fetch relevant memory snippets for the current turn."""
    if not (memory_client and MEMORY_ID and namespace):
        return []

    try:
        records = memory_client.retrieve_memories(
            memory_id=MEMORY_ID,
            namespace=namespace,
            query=query,
            actor_id=actor_id,
            top_k=MEMORY_TOP_K,
        )
        snippets: list[str] = []
        for record in records:
            content = record.get("content", {})
            text = content.get("text") if isinstance(content, dict) else None
            if text:
                snippets.append(text)

        if snippets:
            logger.info(
                "Retrieved %d memory records for session_id=%s namespace=%s",
                len(snippets),
                session_id,
                namespace,
            )
        return snippets
    except Exception as retrieval_error:
        logger.warning(
            "Failed to retrieve memories for session_id=%s namespace=%s: %s",
            session_id,
            namespace,
            retrieval_error,
        )
        return []


def _persist_memory_turn(
    namespace: Optional[str],
    session_id: Optional[str],
    actor_id: Optional[str],
    user_prompt: str,
    assistant_reply: Optional[str],
) -> None:
    """Store the latest conversation turn using AgentCore Memory."""
    if not (memory_client and MEMORY_ID and namespace and session_id and assistant_reply):
        return

    resolved_actor = actor_id or session_id or "default"
    try:
        memory_client.create_event(
            memory_id=MEMORY_ID,
            actor_id=resolved_actor,
            session_id=session_id,
            messages=[
                (user_prompt, "USER"),
                (assistant_reply, "ASSISTANT"),
            ],
        )
        logger.info(
            "Persisted conversation turn to AgentCore Memory for session_id=%s namespace=%s",
            session_id,
            namespace,
        )
    except Exception as persistence_error:
        logger.warning(
            "Failed to persist conversation turn for session_id=%s namespace=%s: %s",
            session_id,
            namespace,
            persistence_error,
        )


def _extract_agent_response(response) -> str:
    """Normalize agent responses across different return types."""
    try:
        if hasattr(response, "message") and response.message:
            return response.message["content"][0]["text"]
        if hasattr(response, "content"):
            return response.content
        if hasattr(response, "text"):
            return response.text
        if hasattr(response, "body"):
            body = response.body
            return body.decode("utf-8") if isinstance(body, bytes) else str(body)
        if "starlette" in str(type(response)):
            if hasattr(response, "_body"):
                body = response._body
                return body.decode("utf-8") if isinstance(body, bytes) else str(body)
            if hasattr(response, "content"):
                return response.content

        result = str(response)
        if "starlette.responses.JSONResponse object" in result:
            logger.error("Failed to extract content from Starlette response: %s", result)
            return "I apologize, but I'm experiencing a technical issue with the response format. Please try again."
        return result
    except Exception as extraction_error:
        logger.error("Error extracting response: %s", extraction_error)
        result = str(response)
        if "starlette.responses.JSONResponse object" in result:
            return "I apologize, but I'm experiencing a technical issue with the response format. Please try again."
        return result

def get_secret(secret_name):
    """
    Retrieve secret from AWS Secrets Manager
    """
    client = boto3.client(
        service_name='secretsmanager'
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        logger.error(f"Error retrieving secret {secret_name}: {e}")
        raise e
    else:
        if 'SecretString' in get_secret_value_response:
            logger.info(f"Successfully retrieved secret {secret_name}")
            return get_secret_value_response['SecretString']

def get_mongo_client():
    mongodb_uri = get_secret("workshop/atlas_secret")  # Replace with your secret name
    logger.info("Creating MongoDB client connection")
    client = MongoClient(mongodb_uri)
    return client


# Initialize Bedrock client and agent with local tools
bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
model = BedrockModel(
    client=bedrock_client,
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0"
)
agent = Agent(
    model=model,
    tools=[current_time, current_month, place_lookup_by_country, place_lookup_by_name, place_best_time_lookup, mongodb_search],
    system_prompt="You are a travel advisor.  You can tell the current time in seconds, or get current month, look up countries by name, look up places to visit and recommend the best time to visit."
)

@app.entrypoint
def run_agent(user_input, context=None) -> str:
    """Run the agent with user input and return response"""
    # Extract the actual prompt from the input
    if isinstance(user_input, dict) and 'prompt' in user_input:
        prompt = user_input['prompt']
    elif isinstance(user_input, str):
        prompt = user_input
    else:
        prompt = str(user_input)

    session_id = getattr(context, "session_id", None) if context else None
    actor_id = None
    if isinstance(user_input, dict):
        actor_id = user_input.get("actorId")
        session_id = session_id or user_input.get("sessionId") or user_input.get("runtimeSessionId")

    logger.info("Processing user input: %s", prompt)

    namespace = _format_memory_namespace(session_id, actor_id)
    memory_snippets: list[str] = []
    if namespace:
        memory_snippets = _retrieve_memory_context(namespace, prompt, session_id, actor_id)

    composed_prompt = prompt
    if memory_snippets:
        memory_block = "\n".join(memory_snippets)
        composed_prompt = f"{MEMORY_PROMPT_PREFIX}{memory_block}\n\nUser: {prompt}"

    response = agent(composed_prompt)

    assistant_reply = _extract_agent_response(response)

    _persist_memory_turn(namespace, session_id, actor_id, prompt, assistant_reply)

    return assistant_reply

if __name__ == "__main__":  
    app.run()
