#!/usr/bin/env python

"""
A script to deploy an Amazon Bedrock Agent Core runtime.

This script provides functionality to deploy and configure an Amazon Bedrock Agent Core runtime
with specified agent name and entry point. It handles the deployment process including:
- Configuring the runtime with required parameters
- Creating execution roles and ECR repositories automatically 
- Launching the runtime
- Monitoring deployment status

The code is built on the Amazon Bedrock Agent Core Starter Toolkit and requires valid AWS credentials.

Dependencies:
- bedrock_agentcore_starter_toolkit
- boto3

Usage:
uv run deploy_to_agentcore.py --agent_name <name> --entry_point <file>

This code has been adapted from:
https://github.com/awslabs/amazon-bedrock-agentcore-samples/blob/main/01-tutorials/01-AgentCore-runtime/01-hosting-agent/01-strands-with-bedrock-model/runtime_with_strands_and_bedrock_models.ipynb
"""

import argparse
import os
import time

from bedrock_agentcore_starter_toolkit import Runtime
from boto3.session import Session

boto_session = Session()
region = os.getenv('AWS_REGION', 'us-west-2')

agentcore_runtime = Runtime()

def wait_for_status():
    """
    Wait for the AgentCore runtime deployment to reach a terminal status.
    
    Polls the runtime status every 10 seconds until one of the following statuses is reached:
    - READY: Deployment completed successfully
    - CREATE_FAILED: Deployment creation failed
    - DELETE_FAILED: Deployment deletion failed  
    - UPDATE_FAILED: Deployment update failed
    
    Prints the current status at each poll interval.
    
    Returns:
        None
    """
    status_response = agentcore_runtime.status()
    status = status_response.endpoint['status']
    end_status = ['READY', 'CREATE_FAILED', 'DELETE_FAILED', 'UPDATE_FAILED']
    while status not in end_status:
        time.sleep(10)
        status_response = agentcore_runtime.status()
        status = status_response.endpoint['status']
        print(status)

def deploy_agentcore(agent_name: str, entry_point: str, requirements_file: str = 'requirements.txt', local_build: bool = False):
    """
    Deploy an Amazon Bedrock Agent Core runtime with the specified configuration.

    Args:
        agent_name (str): Name of the agent to deploy
        entry_point (str): Path to the entry point file for the agent
        requirements_file (str, optional): Path to requirements.txt file. Defaults to 'requirements.txt'
        local_build (bool, optional): Whether to build the container locally. Defaults to False

    Returns:
        Tuple[dict, Runtime]: Tuple containing:
            - Launch result containing deployment status and endpoint information
            - AgentCore runtime instance

    Raises:
        RuntimeError: If configuration or launch fails
        ValueError: If required parameters are invalid

    Example:
        result, runtime = deploy_agentcore(
            agent_name="my-agent",
            entry_point="agent.py"
        )
    """
    response = agentcore_runtime.configure(
        entrypoint=entry_point,
        auto_create_execution_role=True,
        auto_create_ecr=True,
        requirements_file=requirements_file,
        region=region,
        agent_name=agent_name
    )
    launch_result = agentcore_runtime.launch(
        local_build=local_build
    )
    return launch_result, agentcore_runtime


def deploy_agentcore_with_cognito_jwt(agent_name: str, entry_point: str, discovery_url: str, client_id: str, requirements_file: str = 'requirements.txt', local_build: bool = False):
    """
    Deploy an Amazon Bedrock Agent Core runtime with Cognito JWT authorization configuration.

    Args:
        agent_name (str): Name of the agent to deploy
        entry_point (str): Path to the entry point file for the agent
        discovery_url (str): Cognito user pool discovery URL for JWT validation
        client_id (str): Cognito app client ID to allow access
        requirements_file (str, optional): Path to requirements.txt file. Defaults to 'requirements.txt'
        local_build (bool, optional): Whether to build the container locally. Defaults to False

    Returns:
        Tuple[dict, Runtime]: Tuple containing:
            - Launch result containing deployment status and endpoint information
            - AgentCore runtime instance

    Raises:
        RuntimeError: If configuration or launch fails
        ValueError: If required parameters are invalid

    Example:
        result, runtime = deploy_agentcore_with_cognito_jwt(
            agent_name="my-agent",
            entry_point="agent.py",
            discovery_url="https://cognito-idp.region.amazonaws.com/userpool/.well-known/openid-configuration",
            client_id="abc123def456"
        )
    """
    response = agentcore_runtime.configure(
        entrypoint=entry_point,
        auto_create_execution_role=True,
        auto_create_ecr=True,
        requirements_file=requirements_file,
        region=region,
        agent_name=agent_name,
        authorizer_configuration={
            "customJWTAuthorizer": {
                "discoveryUrl": discovery_url,
                "allowedClients": [client_id]
            }
        }
    )
    launch_result = agentcore_runtime.launch(
        local_build=local_build
    )
    return launch_result, agentcore_runtime


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_name', type=str, help='Name of the agent to deploy')
    parser.add_argument('--entry_point', type=str, help='Entry point file for the agent')
    parser.add_argument('--local_build', action='store_true', help='Use local build (only for arm64 platforms)')
    args = parser.parse_args()

    deploy_agentcore(
        agent_name = args.agent_name,
        entry_point = args.entry_point,
        local_build = args.local_build
    )
    wait_for_status()
