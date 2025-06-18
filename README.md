# PostScrypt Memory Collection API
# Warning this is just a demo code for backend. It is not the actual code.
## Overview
PostScrypt is a memory collection system using the OpenAI API with persistent SQL-based conversation storage. It allows users to collect and store memories and stories across various topics.

## Features
- Collect and store conversations based on different topics.
- Retrieve conversation history and statistics.
- Delete specific conversation histories.

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd PostScrypt
   ```

2. **Set Up Environment Variables**
   - Create a `.env` file in the root directory and add the following:
     ```
     OPENAI_API_KEY=your-openai-api-key
     OPENAI_API_URL=https://api.openai.com/v1/chat/completions
     ```
   
   - **On Linux/macOS:**
     ```bash
     export OPENAI_API_KEY=your-openai-api-key
     export OPENAI_API_URL=https://api.openai.com/v1/chat/completions
     ```
   
   - **On Windows (Command Prompt):**
     ```cmd
     set OPENAI_API_KEY=your-openai-api-key
     set OPENAI_API_URL=https://api.openai.com/v1/chat/completions
     ```
   
   - **On Windows (PowerShell):**
     ```powershell
     $env:OPENAI_API_KEY="your-openai-api-key"
     $env:OPENAI_API_URL="https://api.openai.com/v1/chat/completions"
     ```

3. **Install Dependencies**
   - Ensure you have Python 3.7+ installed.
   - Install the required packages:
     ```bash
     pip install -r requirements.txt
     ```

4. **Run the Application**
   - Start the application:
     ```bash
     python main.py
     ```

5. **Access the API**
   - Open your browser and go to `http://localhost:8000/docs` to access the API documentation and test the endpoints.

### API Endpoints
- `POST /chat/query`: Main chat endpoint to interact with the AI.
- `GET /chat/conversations`: Retrieve all conversations.
- `GET /chat/conversation/{person_id}/{topic_id}`: Retrieve conversation history for a specific person and topic.
- `DELETE /chat/conversation/{person_id}/{topic_id}`: Delete conversation history for a specific person and topic.
- `GET /chat/topics`: List available topics and their prompts.
- `GET /health`: Health check endpoint.

## License
This project is licensed under the MIT License. 
