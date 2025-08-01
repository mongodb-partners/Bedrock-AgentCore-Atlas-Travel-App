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
    """Get response from the Bedrock AgentCore."""
    try:
        # Prepare the payload for AgentCore
        payload = json.dumps({"prompt": user_input})
        
        # Invoke the AgentCore runtime
        response = agent_client.invoke_agent_runtime(
            agentRuntimeArn=agent_runtime_arn,
            qualifier="DEFAULT",
            payload=payload
        )
        
        # Extract the response from the streaming body
        response_body = response['response'].read().decode('utf-8')
        
        # Debug: Show what we're getting
        st.write("Debug - Raw response body:", response_body[:200] + "..." if len(response_body) > 200 else response_body)
        
        # Try to parse as JSON first (in case it's wrapped)
        try:
            response_data = json.loads(response_body)
            
            # Debug: Show parsed data type and content
            st.write("Debug - Parsed data type:", type(response_data))
            st.write("Debug - Parsed data:", str(response_data)[:200] + "..." if len(str(response_data)) > 200 else str(response_data))
            
            # Handle the specific case where we get a Starlette object string
            if isinstance(response_data, str) and "starlette.responses.JSONResponse object" in response_data:
                return "I apologize, but I'm experiencing a technical issue with the response format. Please try your question again, or contact support if the issue persists."
            
            # Use str() conversion like app.py does - this handles Starlette objects correctly
            response_str = str(response_data)
            
            # If it's still a Starlette object reference, return an error message
            if "starlette.responses.JSONResponse object" in response_str:
                return "I apologize, but I'm experiencing a technical issue with the response format. Please try your question again."
            
            return response_str
        except json.JSONDecodeError:
            return response_body
            
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
