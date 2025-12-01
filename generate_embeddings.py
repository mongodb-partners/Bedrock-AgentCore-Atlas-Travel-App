import logging
import certifi
import dns.resolver

from pymongo import MongoClient
from langchain_aws.embeddings import BedrockEmbeddings
import boto3
from botocore.exceptions import ClientError

database_name = "travel"
collection_name = "asia"
origin_fields = ["About Place", "Best Time To Visit"]
embedding_field = "details_embedding"

dns.resolver.default_resolver=dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers=['169.254.169.253', '8.8.8.8']

# Configure logging
logging.basicConfig(
    level=logging.WARN,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mdb_import')

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

def setup_bedrock():
    """Initialize the Bedrock runtime."""
    logger.info("Setting up Bedrock runtime client")
    return boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",
    )

# Get the MongoDB connection string from Secrets Manager
logger.info("Retrieving MongoDB connection string from Secrets Manager")
mongodb_uri = get_secret("workshop/atlas_secret")  # Replace with your secret name

# MongoDB connection
logger.info("Connecting to MongoDB Atlas")
client = MongoClient(mongodb_uri, tlsCAFile=certifi.where())

db = client[database_name]
collection = db[collection_name]

bedrock_runtime = setup_bedrock()
bedrock_embeddings = BedrockEmbeddings(
    client=bedrock_runtime,
    model_id="amazon.titan-embed-text-v1", # or another embedding model available in Bedrock such as "amazon.titan-embed-text-v2:0"
)

count = 0
cursor = collection.find()
for document in cursor:
    updates = {}

    text = ""
    # Concatenate origin fields of document into the text variable
    for field in origin_fields:
        field_value = document.get(field, "")
        if field_value:
            adjusted_field_name = field.replace(" ", "_")
            text += f"<{adjusted_field_name}>{field_value}</{adjusted_field_name}>\n"

    if text:
        # Generate embedding using BedrockEmbeddings
        try:
            embedding_vector = bedrock_embeddings.embed_query(text)
            embedding_value = embedding_vector
            updates[embedding_field] = embedding_value
            count += 1
            if (count) % 50 == 0:
                logger.info(f"Generated embedding for {count} documents. ({document['_id']} field '{origin_fields}')")
                print(f"Generated embedding for {count} documents.")
        except Exception as e:
            logger.error(f"Error generating embedding for document ID {document['_id']} field '{origin_fields}': {e}")
    if updates:
        collection.update_one({'_id': document['_id']}, {'$set': updates})
        logger.info(f"Updated document ID {document['_id']} with embeddings")

logger.info(f"Total documents processed: {count}")
print(f"Total documents processed: {count}")
