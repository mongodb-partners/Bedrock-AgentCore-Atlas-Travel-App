import streamlit as st
import boto3
import json

# Set page configuration
st.set_page_config(
    page_title="MongoDB Atlas Travel Assistant (AgentCore)",
    page_icon="🌍",
    layout="wide"
)

# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

def initialize_bedrock_agent_client():
    """Initialize the Bedrock AgentCore client."""
    return boto3.client(
        service_name="bedrock-agentcore",
        region_name="us-east-1"
    )

def get_agent_response(agent_client, agent_runtime_arn, user_input):
    """Get response from the Bedrock AgentCore using the same logic as app.py."""
    try:
        # Prepare the payload for AgentCore
        payload = json.dumps({"prompt": user_input})
        
        # Invoke the AgentCore runtime
        response = agent_client.invoke_agent_runtime(
            agentRuntimeArn=agent_runtime_arn,
            qualifier="DEFAULT",
            payload=payload
        )
        
        # Extract response using the same method as app.py
        raw_content = response['response'].read()
        decoded_content = raw_content.decode('utf-8')
        
        # Parse the JSON-encoded response to get the actual text
        try:
            agent_response = json.loads(decoded_content)
            
            # Check if we got a Starlette object string instead of actual content
            if isinstance(agent_response, str) and "starlette.responses.JSONResponse object" in agent_response:
                return "⚠️ I'm experiencing a technical issue with the response format. The agent is working but there's a serialization problem. Please try your question again."
            
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
            value="<AGENT-ARN>"
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
                # Initialize Bedrock agent client
                agent_client = initialize_bedrock_agent_client()
                
                # Get response from Bedrock agent
                response = get_agent_response(agent_client, agent_runtime_arn, prompt)
                
                # Display the response
                st.write(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
