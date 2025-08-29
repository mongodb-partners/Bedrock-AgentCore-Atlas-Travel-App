"""
A script to interact with Amazon Bedrock Agent runtimes.

This script provides functionality to:
- List available agent runtimes
- Invoke an agent runtime with a prompt
- Process and display the agent's response

Requires:
    - boto3
    - Valid AWS credentials with appropriate permissions
    - AWS_REGION environment variable (defaults to us-west-2)
"""

import argparse
import boto3
import json
import os


region = os.getenv('AWS_REGION', 'us-west-2')
print(f'Using region: {region}')

agentcore_client = boto3.client(
    'bedrock-agentcore',
    region_name=region
)
agentcore_control_client = boto3.client(
    'bedrock-agentcore-control',
    region_name=region
)


def get_agent_runtimes():
    response = agentcore_control_client.list_agent_runtimes()
    runtimes = [ runtime for runtime in response['agentRuntimes'] ]
    print('-' * 80)
    for runtime in runtimes:
        print(f"Agent Name: {runtime['agentRuntimeName']}")
        print(f"ARN: {runtime['agentRuntimeArn']}")
        print('-' * 80)
    return runtimes


def invoke_agent_runtime(agent_arn, payload):
    """
    Invokes an Amazon Bedrock Agent runtime with the specified ARN and payload.

    Args:
        agent_arn (str): The ARN of the agent runtime to invoke
        payload (str): JSON payload containing the prompt or other parameters

    Returns:
        dict: Response from the agent runtime invocation

    Reference:
    - https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-invoke-agent.html
    - https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/response-streaming.html
    """

    response = agentcore_client.invoke_agent_runtime(
        agentRuntimeArn=agent_arn,
        qualifier="DEFAULT",
        payload=payload
    )
    if "text/event-stream" in response.get("contentType", ""):
        content = []
        for line in response["response"].iter_lines(chunk_size=10):
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    line = line[6:]
                    try:
                        data = json.loads(line)
                        if isinstance(data, dict):
                            event = data.get('event', '')
                            contentBlockDelta = event.get('contentBlockDelta', '')
                            delta = contentBlockDelta.get('delta', '')
                            text = delta.get('text', '')
                            content.append(text)
                            print(text, end='')
                    except Exception as e:
                        pass
                    # print(line)
                    # content.append(line)
        return ''.join(content)

    elif response.get("contentType") == "application/json":
        # Handle standard JSON response
        content = []
        for chunk in response.get("response", []):
            content.append(chunk.decode('utf-8'))
        print(json.loads(''.join(content)))
        return '\n'.join(content)
    else:
        # Print raw response for other content types
        print(response)
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Invoke an agent runtime with a prompt')
    parser.add_argument('--prompt', type=str, default="What is the weather like in Seattle?",
                        help='The prompt to send to the agent runtime')
    args = parser.parse_args()

    runtimes = get_agent_runtimes()
    if len(runtimes):
        agent_arn = runtimes[0]['agentRuntimeArn']
        payload = json.dumps({"prompt": args.prompt})
        print(f'Invoking agent with payload:\n{payload}\n')
        response = invoke_agent_runtime(agent_arn, payload)
        print()
