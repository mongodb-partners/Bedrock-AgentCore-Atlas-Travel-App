import os
import uuid
import json

import requests
import streamlit as st
import boto3

# Set page configuration
st.set_page_config(
    page_title="MongoDB Atlas Travel Assistant (AgentCore)",
    page_icon="🌍",
    layout="wide"
)

# Local AgentCore endpoint configuration
LOCAL_AGENT_ENDPOINT = os.getenv("LOCAL_AGENTCORE_URL", "http://127.0.0.1:8080/invocations")
DEFAULT_AGENT_PLACEHOLDER = "<AGENT-ARN>"

# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = f"streamlit-{uuid.uuid4().hex[:8]}"
if "actor_id" not in st.session_state:
    st.session_state.actor_id = "streamlit-user"

def initialize_bedrock_agent_client():
    """Initialize the Bedrock AgentCore client."""
    return boto3.client(
        service_name="bedrock-agentcore",
        region_name="us-east-1"
    )

def get_agent_response(agent_client, agent_runtime_arn, user_input):
    """Get response from Bedrock AgentCore or local runtime."""
    try:
        payload_dict = {
            "prompt": user_input,
            "sessionId": st.session_state.session_id,
            "actorId": st.session_state.actor_id,
        }

        if not agent_runtime_arn or agent_runtime_arn.strip() in {"", DEFAULT_AGENT_PLACEHOLDER}:
            response = requests.post(
                LOCAL_AGENT_ENDPOINT,
                json=payload_dict,
                timeout=90,
            )
            response.raise_for_status()
            try:
                return response.json()
            except json.JSONDecodeError:
                return response.text

        payload = json.dumps(payload_dict)
        response = agent_client.invoke_agent_runtime(
            agentRuntimeArn=agent_runtime_arn,
            qualifier="DEFAULT",
            payload=payload,
        )

        raw_content = response["response"].read()
        decoded_content = raw_content.decode("utf-8")

        try:
            agent_response = json.loads(decoded_content)
            if (
                isinstance(agent_response, str)
                and "starlette.responses.JSONResponse object" in agent_response
            ):
                return (
                    "⚠️ I'm experiencing a technical issue with the response format. "
                    "The agent is working but there's a serialization problem. "
                    "Please try your question again."
                )
            return agent_response
        except json.JSONDecodeError:
            return decoded_content

    except Exception as e:
        st.error(f"Error invoking AgentCore: {str(e)}")
        return f"Error: {str(e)}"

# Main app
def main():
    # App header
    st.title("🌍 MongoDB Atlas Travel Assistant")
    st.markdown("""
    Ask questions about travel destinations, best times to visit, and more!
    """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        agent_runtime_arn = st.text_input(
            "Agent Runtime ARN",
            value=DEFAULT_AGENT_PLACEHOLDER,
        )
        
        # Add a clear chat button
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about travel destinations..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                use_remote = agent_runtime_arn.strip() not in {"", DEFAULT_AGENT_PLACEHOLDER}
                agent_client = initialize_bedrock_agent_client() if use_remote else None

                response = get_agent_response(agent_client, agent_runtime_arn, prompt)
                
                # Display the response
                st.write(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
