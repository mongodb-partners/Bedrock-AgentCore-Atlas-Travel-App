import streamlit as st
import json
from invoke import agentcore_client, agentcore_control_client, get_agent_runtimes, invoke_agent_runtime, region

# Set page configuration
st.set_page_config(
    page_title="MongoDB Atlas Travel Assistant (AgentCore)",
    page_icon="🌍",
    layout="wide"
)

# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize session state for agent runtimes if it doesn't exist
if "agent_runtimes" not in st.session_state:
    st.session_state.agent_runtimes = []

@st.cache_data
def load_agent_runtimes():
    """Load available agent runtimes using invoke.py function."""
    try:
        return get_agent_runtimes()
    except Exception as e:
        st.error(f"Error loading agent runtimes: {str(e)}")
        return []

def get_agent_response(agent_runtime_arn, user_input):
    """Get response from the Bedrock AgentCore using invoke.py function."""
    try:
        # Prepare the payload for AgentCore
        payload = json.dumps({"prompt": user_input})
        
        # Use the invoke_agent_runtime function from invoke.py
        response = invoke_agent_runtime(agent_runtime_arn, payload)
        
        return response
            
    except Exception as e:
        st.error(f"Error invoking AgentCore: {str(e)}")
        return f"Error: {str(e)}"

# Main app
def main():
    # App header
    st.title("🌍 MongoDB Atlas Travel Assistant")
    st.markdown(f"""
    Ask questions about travel destinations, best times to visit, and more!
    
    **Region:** {region}
    """)
    
    # Load agent runtimes
    if not st.session_state.agent_runtimes:
        with st.spinner("Loading available agent runtimes..."):
            st.session_state.agent_runtimes = load_agent_runtimes()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Agent runtime selection
        if st.session_state.agent_runtimes:
            # Create options for the selectbox
            runtime_options = []
            runtime_arns = []
            
            for runtime in st.session_state.agent_runtimes:
                runtime_name = runtime.get('agentRuntimeName', 'Unknown')
                runtime_arn = runtime.get('agentRuntimeArn', '')
                runtime_options.append(f"{runtime_name}")
                runtime_arns.append(runtime_arn)
            
            if runtime_options:
                selected_index = st.selectbox(
                    "Select Agent Runtime:",
                    range(len(runtime_options)),
                    format_func=lambda x: runtime_options[x],
                    help="Choose from available agent runtimes"
                )
                agent_runtime_arn = runtime_arns[selected_index]
                
                # Display selected ARN for reference
                st.text_area(
                    "Selected ARN:",
                    value=agent_runtime_arn,
                    height=100,
                    disabled=True,
                    help="This is the ARN of the selected agent runtime"
                )
            else:
                st.error("No agent runtimes found")
                agent_runtime_arn = None
        else:
            st.error("Failed to load agent runtimes")
            agent_runtime_arn = None
        
        # Refresh runtimes button
        if st.button("🔄 Refresh Agent Runtimes"):
            st.session_state.agent_runtimes = []
            st.cache_data.clear()
            st.rerun()
        
        # Add a clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Check if we have a valid agent runtime ARN
    if not agent_runtime_arn:
        st.warning("⚠️ Please select an agent runtime from the sidebar to start chatting.")
        return
    
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
                # Get response from Bedrock agent using invoke.py function
                response = get_agent_response(agent_runtime_arn, prompt)
                
                # Display the response (replace literal \n with actual newlines)
                formatted_response = response.replace('\\n\\n', '\n\n').replace('\\n', '\n')
                st.write(formatted_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
