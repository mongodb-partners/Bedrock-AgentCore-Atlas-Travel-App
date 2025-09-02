import csv
import logging
from pymongo import MongoClient
import boto3
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
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

# Get the MongoDB connection string from Secrets Manager
logger.info("Retrieving MongoDB connection string from Secrets Manager")
mongodb_uri = get_secret("workshop/atlas_secret")  # Replace with your secret name

# MongoDB connection
logger.info("Connecting to MongoDB Atlas")
client = MongoClient(mongodb_uri)

db = client['travel']
collection = db['asia']

# CSV file path
csv_file_path = './anthropic-travel-agency.trip_recommendations.csv'

logger.info('Starting data import from CSV to MongoDB')

index = 1
with open(csv_file_path, mode='r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # logger.debug(f"Processing row type: {type(row)}")
        row['index'] = index
        index += 1
        # loop over columns and accumulate all detail_embedding into an array
        detail_embedding = []
        new_row = {}
        for column in row.keys():
            if column.startswith('details_embedding'):
                detail_embedding.append(float(row[column]))
            else:
                new_row[column] = row[column]
        
        new_row['details_embedding'] = detail_embedding

        collection.insert_one(new_row)
        if index % 25 == 0:
            logger.info(f'Inserted {index} rows')

logger.info('Finished import successfully')
