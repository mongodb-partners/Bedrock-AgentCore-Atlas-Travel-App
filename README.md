# Bedrock AgentCore Travel Assistant

A travel assistant application built with AWS Bedrock AgentCore and MongoDB Atlas integration, providing comprehensive travel information and recommendations.

## Overview

This project demonstrates a modern AI-powered travel assistant that leverages:
- **AWS Bedrock AgentCore** for intelligent agent orchestration
- **MongoDB Atlas** for travel data storage and retrieval
- **Streamlit** for interactive web interface
- **Docker** for containerized deployment

The agent provides detailed travel information including destinations, activities, cultural insights, and practical travel advice with graceful fallback when database access is unavailable.

## Project Structure

```
mdb_strands_agentcore/
├── agent.py                    # Core agent logic and MongoDB integration
├── app.py                      # Direct API testing interface
├── streamlit_app.py           # Streamlit web interface
├── deploy.py                  # AWS deployment script with IAM configuration
├── requirements.txt           # Python dependencies
├── venv/                     # Python virtual environment
└── README.md                 # This file
```

## Features

- **Intelligent Travel Recommendations**: Provides comprehensive travel advice for destinations worldwide
- **MongoDB Atlas Integration**: Stores and retrieves travel data with graceful fallback
- **Multi-Interface Support**: Both direct API and Streamlit web interface
- **Robust Error Handling**: Continues to provide valuable responses even when data sources are unavailable
- **AWS Integration**: Full deployment pipeline with proper IAM permissions
- **Containerized Deployment**: Docker support for consistent environments

## Prerequisites

- Python 3.12+
- AWS CLI configured with appropriate permissions
- MongoDB Atlas account
- Docker (for containerized deployment)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mongodb-partners/Bedrock-AgentCore-Atlas-Travel-App.git
   ```

2. **Set up virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure AWS credentials**:
   ```bash
   aws configure
   ```

## Usage

### Direct API Testing

Test the agent directly using the command-line interface:

```bash
source venv/bin/activate
python app.py
```

This will run a test query and display the agent's response with debug information.

### Streamlit Web Interface

Launch the interactive web interface:

```bash
source venv/bin/activate
streamlit run streamlit_app.py
```

Navigate to `http://localhost:8501` to interact with the travel assistant through a user-friendly web interface.

## Configuration

**Important**: Replace `<AGENT-ARN>` in the code with your actual Bedrock AgentCore ARN.

After running `python deploy.py`, use the ARN from the output in:
- `app.py`: Update the `agentRuntimeArn` parameter
- `streamlit_app.py`: Update the agent ARN reference

Example:
```python
agentRuntimeArn="arn:aws:bedrock-agentcore:us-east-1:123456789:runtime/agentcore_name-id"
```

## Deployment

Deploy the agent to AWS using the deployment script:

```bash
source venv/bin/activate
python deploy.py
```

The deployment script:
- Creates necessary IAM roles and policies
- Configures Secrets Manager access for MongoDB credentials
- Deploys the agent to Bedrock AgentCore
- Sets up proper permissions for all AWS services

## Architecture

### Core Components

1. **Agent Core (`agent.py`)**:
   - Handles travel queries and recommendations
   - Integrates with MongoDB Atlas for data retrieval
   - Implements graceful fallback mechanisms

2. **Simple Test App (`app.py`)**:
   - Direct API testing and debugging
   - Response extraction and formatting
   - Dynamic agent ARN lookup from configuration

3. **Web Interface (`streamlit_app.py`)**:
   - User-friendly chat interface
   - Real-time agent interaction
   - Response parsing and display

4. **Deployment (`deploy.py`)**:
   - AWS resource provisioning
   - IAM role and policy management
   - Secrets Manager integration

### Data Flow

```
User Query → Streamlit/API → Bedrock AgentCore → Agent Logic → MongoDB Atlas
                                                      ↓
User Interface ← Response Processing ← Agent Response ← Travel Data/Fallback
```

## Troubleshooting

### Common Issues

1. **400 Bad Request Errors**:
   - Check input parameter formatting in agent calls
   - Verify agent ARN is correctly configured

2. **AccessDeniedException for Secrets Manager**:
   - Ensure IAM role has `secretsmanager:GetSecretValue` permission
   - Verify the secret exists and is accessible

3. **MongoDB Connection Issues**:
   - Agent provides comprehensive fallback responses
   - Verify MongoDB Atlas credentials in Secrets Manager

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly using both interfaces
5. Submit a pull request

## License

[Add your license information here]

## Support

For issues and questions:
- Check the troubleshooting section above
- Review CloudWatch logs for detailed error information
- Ensure all AWS permissions are properly configured

---

**Note**: This project demonstrates advanced AWS Bedrock AgentCore integration with MongoDB Atlas, showcasing modern AI application architecture with robust error handling and multiple interface options.
