# FinAssist 🤖

FinAssist is an AI-powered chatbot designed to assist users with financial support and guidance. Built using LangChain, Groq LLM, and Streamlit, this app provides intelligent, personalized conversations about various financial topics. 

## Features
- AI-powered chatbot to provide instant financial support.
- Real-time responses using the Groq language model.
- User-friendly and visually appealing interface.
- History of past interactions, easily accessible during sessions.
- Option to clear the chat history for a fresh start.

## Tech Stack
- **Streamlit:** Framework for building interactive web apps.
- **LangChain & Groq LLM:** For large language model integration.
- **Python:** Programming language for backend logic and model invocation.
- **HTML/CSS:** For customizing the frontend to provide a modern and clean user interface.
- **dotenv:** Used to manage environment variables like API keys securely.

## Getting Started

To run this application locally, follow the steps below:

### Prerequisites

- Python 3.7 or higher
- Virtual environment (recommended)

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/viswa2k4/Digital_Assignment_3_Viswanathan.git
    cd Digital_Assignment_3_Viswanathan
    ```

2. Set up a virtual environment (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Create a `.env` file in the root directory of the project and add your Groq API key:
    ```bash
    GROQ_API_KEY="your-api-key-here"
    ```

### Run the Application

After setting up the project and installing dependencies, run the app with the following command:

```bash
streamlit run app.py
