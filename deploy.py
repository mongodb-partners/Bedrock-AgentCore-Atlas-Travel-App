"""Deployment helper for Bedrock AgentCore travel assistant."""

import json
import os
import re
import sys
import time
import traceback
from typing import Optional

import boto3
from botocore.exceptions import ClientError
from boto3.session import Session

from bedrock_agentcore.memory import MemoryClient
from bedrock_agentcore.memory.constants import StrategyType


session = Session()
region = session.region_name or "us-east-1"

# Deployment configuration (override with environment variables)
AGENT_NAME = os.getenv("AGENTCORE_AGENT_NAME", "atlas_travel_agent")
CONTAINER_IMAGE_URI = os.getenv("AGENTCORE_IMAGE_URI", "979559056307.dkr.ecr.us-east-1.amazonaws.com/agentcore-travel-app-anuj:latest")


def _sanitize_memory_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if not cleaned:
        cleaned = "memory"
    if not cleaned[0].isalpha():
        cleaned = f"m_{cleaned}"
    return cleaned[:48]


MEMORY_NAME = _sanitize_memory_name(os.getenv("AGENTCORE_MEMORY_NAME", f"{AGENT_NAME}_memory"))
MEMORY_NAMESPACE = os.getenv("AGENTCORE_MEMORY_NAMESPACE", "/travel/{sessionId}")
MEMORY_TOP_K = os.getenv("AGENTCORE_MEMORY_TOP_K", "5")
MEMORY_PROMPT_PREFIX = os.getenv(
    "AGENTCORE_MEMORY_PROMPT_PREFIX",
    "Here are details from our recent conversation that may help:\n",
)
MEMORY_EXECUTION_ROLE_ARN = os.getenv("AGENTCORE_MEMORY_ROLE_ARN")


def create_agentcore_role(agent_name: str) -> str:
    """Create (or recreate) the execution role used by AgentCore runtime."""
    iam_client = boto3.client("iam", region_name=region)
    agentcore_role_name = f"agentcore-{agent_name}-role"
    account_id = boto3.client("sts").get_caller_identity()["Account"]
    role_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "BedrockPermissions",
                "Effect": "Allow",
                "Action": [
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream"
                ],
                "Resource": "*"
            },
            {
                "Sid": "ECRImageAccess",
                "Effect": "Allow",
                "Action": [
                    "ecr:BatchGetImage",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:GetAuthorizationToken",
                    "ecr:BatchGetImage",
                    "ecr:GetDownloadUrlForLayer"
                ],
                "Resource": [
                    f"arn:aws:ecr:{region}:{account_id}:repository/*"
                ]
            },
            {
                "Effect": "Allow",
                "Action": [
                    "logs:DescribeLogStreams",
                    "logs:CreateLogGroup"
                ],
                "Resource": [
                    f"arn:aws:logs:{region}:{account_id}:log-group:/aws/bedrock-agentcore/runtimes/*"
                ]
            },
            {
                "Effect": "Allow",
                "Action": [
                    "logs:DescribeLogGroups"
                ],
                "Resource": [
                    f"arn:aws:logs:{region}:{account_id}:log-group:*"
                ]
            },
            {
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogStream",
                    "logs:PutLogEvents"
                ],
                "Resource": [
                    f"arn:aws:logs:{region}:{account_id}:log-group:/aws/bedrock-agentcore/runtimes/*:log-stream:*"
                ]
            },
            {
                "Sid": "ECRTokenAccess",
                "Effect": "Allow",
                "Action": [
                    "ecr:GetAuthorizationToken"
                ],
                "Resource": "*"
            },
            {
            "Effect": "Allow",
            "Action": [
                "xray:PutTraceSegments",
                "xray:PutTelemetryRecords",
                "xray:GetSamplingRules",
                "xray:GetSamplingTargets"
                ],
             "Resource": [ "*" ]
             },
             {
                "Effect": "Allow",
                "Resource": "*",
                "Action": "cloudwatch:PutMetricData",
                "Condition": {
                    "StringEquals": {
                        "cloudwatch:namespace": "bedrock-agentcore"
                    }
                }
            },
            {
                "Sid": "GetAgentAccessToken",
                "Effect": "Allow",
                "Action": [
                    "bedrock-agentcore:GetWorkloadAccessToken",
                    "bedrock-agentcore:GetWorkloadAccessTokenForJWT",
                    "bedrock-agentcore:GetWorkloadAccessTokenForUserId"
                ],
                "Resource": [
                  f"arn:aws:bedrock-agentcore:{region}:{account_id}:workload-identity-directory/default",
                  f"arn:aws:bedrock-agentcore:{region}:{account_id}:workload-identity-directory/default/workload-identity/{agent_name}-*"
                ]
            },
            {
                "Sid": "SecretsManagerAccess",
                "Effect": "Allow",
                "Action": [
                    "secretsmanager:GetSecretValue"
                ],
                "Resource": [
                    f"arn:aws:secretsmanager:{region}:{account_id}:secret:workshop/atlas_secret*"
                ]
            }
        ]
    }
    assume_role_policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "AssumeRolePolicy",
                "Effect": "Allow",
                "Principal": {
                    "Service": "bedrock-agentcore.amazonaws.com"
                },
                "Action": "sts:AssumeRole",
                "Condition": {
                    "StringEquals": {
                        "aws:SourceAccount": f"{account_id}"
                    },
                    "ArnLike": {
                        "aws:SourceArn": f"arn:aws:bedrock-agentcore:{region}:{account_id}:*"
                    }
                }
            }
        ]
    }

    assume_role_policy_document_json = json.dumps(
        assume_role_policy_document
    )
    role_policy_document = json.dumps(role_policy)
    # Create IAM Role for the Lambda function
    try:
        iam_client.create_role(
            RoleName=agentcore_role_name,
            AssumeRolePolicyDocument=assume_role_policy_document_json,
        )
        # Pause briefly to ensure the role propagates
        time.sleep(10)
        print(f"Created IAM role {agentcore_role_name}")
    except iam_client.exceptions.EntityAlreadyExistsException:
        print(f"IAM role {agentcore_role_name} already exists; updating inline policy")
    except ClientError as error:
        raise RuntimeError(f"Failed to create IAM role: {error}") from error

    try:
        iam_client.put_role_policy(
            PolicyDocument=role_policy_document,
            PolicyName="AgentCorePolicy",
            RoleName=agentcore_role_name,
        )
    except ClientError as error:
        raise RuntimeError(f"Failed to attach inline policy: {error}") from error

    role = iam_client.get_role(RoleName=agentcore_role_name)
    role_arn = role["Role"]["Arn"]
    print(f"AgentCore runtime role ARN: {role_arn}")
    return role_arn


def ensure_memory(memory_name: str, namespace: str) -> str:
    """Create or reuse a memory resource and return its ID."""
    memory_client = MemoryClient(region_name=region)

    try:
        existing_memories = memory_client.list_memories()
        print(existing_memories)
    except ClientError as error:
        raise RuntimeError(f"Failed to list memories: {error}") from error

    for memory in existing_memories:
        if memory_name in memory.get("memoryId"):
            memory_id = memory.get("memoryId") or memory.get("id")
            print(memory_id)
            if memory_id:
                print(f"Reusing memory '{memory_name}' (ID: {memory_id})")
                return memory_id

    strategies = [
        {
            StrategyType.SEMANTIC.value: {
                "name": "travelContext",
                "namespaces": [namespace],
            }
        }
    ]

    try:
        result = memory_client.create_memory_and_wait(
            name=memory_name,
            strategies=strategies,
            description="Short-term conversation memory for the travel assistant",
            memory_execution_role_arn=MEMORY_EXECUTION_ROLE_ARN,
        )
        memory_id = result.get("memoryId") or result.get("id")
        if not memory_id:
            raise RuntimeError("Memory creation response did not include an ID")
        print(f"Created memory '{memory_name}' (ID: {memory_id})")
        return memory_id
    except ClientError as error:
        raise RuntimeError(f"Failed to create memory: {error}") from error

## Commenting out wait function for now
# def wait_for_runtime_status(client, runtime_id: str, desired_status: str = "ACTIVE", timeout: int = 900):
#     """Wait until the agent runtime reaches the desired status or fails."""
#     start = time.time()
#     while time.time() - start < timeout:
#         response = client.get_agent_runtime(agentRuntimeId=runtime_id)
#         runtime = response.get("agentRuntime") or response.get("agentRuntimeSummary")
#         if runtime is None:
#             raise RuntimeError(
#                 f"Unexpected response while polling runtime {runtime_id}: {json.dumps(response, default=str)}"
#             )
#         status = runtime.get("status")
#         if status == desired_status:
#             return runtime
#         if status in {"FAILED", "DELETING"}:
#             raise RuntimeError(f"Runtime {runtime_id} entered terminal status: {status}")
#         time.sleep(10)
#     raise TimeoutError(f"Runtime {runtime_id} did not reach {desired_status} within {timeout} seconds")


def create_or_update_runtime(role_arn: str, memory_id: str) -> dict:
    """Create or update an AgentCore runtime with memory environment variables."""
    if not CONTAINER_IMAGE_URI:
        raise RuntimeError(
            "Container image URI not provided. Set AGENTCORE_IMAGE_URI to your ECR image (account.dkr.ecr.region.amazonaws.com/repo:tag)."
        )

    control_client = boto3.client("bedrock-agentcore-control", region_name=region)
    environment_variables = {
        "MEMORY_ID": memory_id,
        "MEMORY_NAMESPACE": MEMORY_NAMESPACE,
        "MEMORY_TOP_K": MEMORY_TOP_K,
        "MEMORY_PROMPT_PREFIX": MEMORY_PROMPT_PREFIX,
    }

    artifact = {"containerConfiguration": {"containerUri": CONTAINER_IMAGE_URI}}
    network_configuration = {"networkMode": "PUBLIC"}
    protocol_configuration = {"serverProtocol": "HTTP"}

    try:
        runtimes = control_client.list_agent_runtimes()["agentRuntimes"]
    except ClientError as error:
        raise RuntimeError(f"Failed to list agent runtimes: {error}") from error

    target_runtime: Optional[dict] = None
    for runtime in runtimes:
        if runtime.get("agentRuntimeName") == AGENT_NAME or runtime.get("name") == AGENT_NAME:
            target_runtime = runtime
            break

    if target_runtime:
        runtime_id = target_runtime["agentRuntimeId"]
        print(f"Updating existing AgentCore runtime {AGENT_NAME} ({runtime_id})")
        try:
            control_client.update_agent_runtime(
                agentRuntimeId=runtime_id,
                agentRuntimeArtifact=artifact,
                roleArn=role_arn,
                networkConfiguration=network_configuration,
                protocolConfiguration=protocol_configuration,
                environmentVariables=environment_variables,
            )
        except ClientError as error:
            raise RuntimeError(f"Failed to update agent runtime: {error}") from error
    else:
        print(f"Creating AgentCore runtime {AGENT_NAME}")
        try:
            response = control_client.create_agent_runtime(
                agentRuntimeName=AGENT_NAME,
                agentRuntimeArtifact=artifact,
                roleArn=role_arn,
                networkConfiguration=network_configuration,
                protocolConfiguration=protocol_configuration,
                environmentVariables=environment_variables,
            )
            print("\n")
            print(response)
            print("\n")
        except ClientError as error:
            raise RuntimeError(f"Failed to create agent runtime: {error}") from error
        runtime_id = response["agentRuntimeId"]

    ## Uncomment below to wait for runtime to be active
    # runtime = wait_for_runtime_status(control_client, runtime_id)
    # print(f"Runtime status: {runtime.get('status')}")
    return response


def main():
    print(f"Deploying to region: {region}")

    role_arn = create_agentcore_role(AGENT_NAME)
    memory_id = ensure_memory(MEMORY_NAME, MEMORY_NAMESPACE)
    runtime = create_or_update_runtime(role_arn, memory_id)

    runtime_arn = runtime.get("agentRuntimeArn")
    endpoint_hint = runtime.get("endpoint") or runtime.get("endpointArn") or "Use create_agent_runtime_endpoint to expose externally."

    print("")
    print("Deployment complete ✨")
    print(f"Agent runtime ARN: {runtime_arn}")
    print(f"Memory ID: {memory_id}")
    print(f"Runtime endpoint info: {endpoint_hint}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Deployment failed: {exc}")
        sys.exit(1)
