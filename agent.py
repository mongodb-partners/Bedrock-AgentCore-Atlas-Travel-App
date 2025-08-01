import time
import logging
import boto3

from pymongo import MongoClient
from langchain_aws.embeddings import BedrockEmbeddings
from botocore.exceptions import ClientError
from strands import Agent, tool
from strands.models import BedrockModel
from bedrock_agentcore.runtime import BedrockAgentCoreApp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = BedrockAgentCoreApp()

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
def run_agent(user_input) -> str:
    """Run the agent with user input and return response"""
    # Extract the actual prompt from the input
    if isinstance(user_input, dict) and 'prompt' in user_input:
        prompt = user_input['prompt']
    elif isinstance(user_input, str):
        prompt = user_input
    else:
        prompt = str(user_input)
    
    logger.info(f"Processing user input: {prompt}")
    response = agent(prompt)
    
    # Handle different response types
    try:
        # Try to get the content from message structure
        if hasattr(response, 'message') and response.message:
            return response.message['content'][0]['text']
        # Try to get content attribute
        elif hasattr(response, 'content'):
            return response.content
        # Try to get text attribute
        elif hasattr(response, 'text'):
            return response.text
        # Handle Starlette JSONResponse objects specifically
        elif hasattr(response, 'body'):
            if isinstance(response.body, bytes):
                return response.body.decode('utf-8')
            else:
                return str(response.body)
        # Check if it's a Starlette JSONResponse and try to get the body
        elif str(type(response)).find('starlette') != -1:
            # For Starlette responses, try to access the body directly
            if hasattr(response, '_body'):
                body = response._body
                if isinstance(body, bytes):
                    return body.decode('utf-8')
                else:
                    return str(body)
            # If no _body, try other Starlette-specific attributes
            elif hasattr(response, 'content'):
                return response.content
            else:
                logger.warning(f"Starlette response detected but couldn't extract content: {type(response)}")
                return "I apologize, but I'm experiencing a technical issue with the response format. Please try again."
        # Fallback to string conversion
        else:
            result = str(response)
            # If we get a Starlette object string, return an error message
            if "starlette.responses.JSONResponse object" in result:
                logger.error(f"Failed to extract content from Starlette response: {result}")
                return "I apologize, but I'm experiencing a technical issue with the response format. Please try again."
            return result
    except Exception as e:
        logger.error(f"Error extracting response: {e}")
        result = str(response)
        if "starlette.responses.JSONResponse object" in result:
            return "I apologize, but I'm experiencing a technical issue with the response format. Please try again."
        return result

if __name__ == "__main__":  
    app.run()