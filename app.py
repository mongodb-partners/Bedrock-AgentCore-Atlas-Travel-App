import boto3
import json

# Initialize the Bedrock AgentCore client
client = boto3.client('bedrock-agentcore', region_name='us-east-1')

input_text = "What places can I visit in India?"
account_id = boto3.client("sts").get_caller_identity()["Account"]

response = client.invoke_agent_runtime(
    agentRuntimeArn="<AGENT-ARN>",
    qualifier="DEFAULT",
    payload=json.dumps({"prompt": input_text})
)

# Extract the response text from the streaming body
response_body = response['response'].read().decode('utf-8')

print("Response Status Code:", response['statusCode'])
print("Raw Response Body:")
print(response_body)

# Enhanced response extraction logic to handle Starlette JSONResponse objects
def extract_response_content(response_body):
    """
    Extract actual content from various response formats including Starlette JSONResponse objects
    """
    try:
        # First, try to parse as JSON
        response_data = json.loads(response_body)
        
        # If it's a string that looks like a Starlette object, we need different handling
        if isinstance(response_data, str) and "starlette.responses.JSONResponse object" in response_data:
            print("DEBUG: Detected Starlette JSONResponse object string")
            return "Response is wrapped in Starlette JSONResponse object - content extraction needed"
        
        # If it's a dict, look for common response fields
        if isinstance(response_data, dict):
            # Try common response field names
            for field in ['content', 'text', 'response', 'message', 'body']:
                if field in response_data:
                    return response_data[field]
            
            # If no common fields, return the whole dict formatted nicely
            return json.dumps(response_data, indent=2)
        
        # If it's already a string, return it
        return str(response_data)
        
    except json.JSONDecodeError:
        # If it's not JSON, check if it's a Starlette object string representation
        if "starlette.responses.JSONResponse object" in response_body:
            print("DEBUG: Raw Starlette JSONResponse object detected")
            return "Raw Starlette JSONResponse object - needs proper extraction method"
        
        # Otherwise return as-is
        return response_body

# Try to extract the actual response content
try:
    print("\nExtracting response content...")
    extracted_content = extract_response_content(response_body)
    print("\nAgent Response:")
    print(extracted_content)
    
    # Additional debug information
    print(f"\nDEBUG INFO:")
    print(f"Response type: {type(response_body)}")
    print(f"Response length: {len(response_body)}")
    print(f"Contains 'starlette': {'starlette' in response_body.lower()}")
    
except Exception as e:
    print(f"Error extracting response: {e}")
    print("Raw response body:", response_body)