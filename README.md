# Legal Edge AI

Legal Edge AI is an AI-powered legal analysis tool that leverages **Neo4j, knowledge graphs, and Retrieval-Augmented Generation (RAG)** to enhance tax and legal regulation insights. By integrating **LangChain, LangGraph, and Streamlit**, it enables intelligent, graph-based retrieval and compliance analysis.

## Features
- **Graph-Based Retrieval**: Uses Neo4j for efficient legal document search and relationship mapping.
- **LLM-Powered Analysis**: Enhances legal insights using advanced Large Language Models (LLMs).
- **Regulatory Compliance**: Assists in understanding tax and legal regulations.
- **Streamlit UI**: Provides an interactive and user-friendly legal research experience.

## Tech Stack
- **Neo4j** – Graph database for structured legal knowledge.
- **LangChain & LangGraph** – AI-powered document processing and retrieval.
- **Streamlit** – Web-based UI for legal insights.
- **Python** – Core development language.

## Installation
```bash
# Clone the repository
git clone https://github.com/etisamhaq/Legal-Edge-AI
cd LexGraph-AI

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
```bash
# Run the application
streamlit run app.py
```

## Configuration
- Ensure **Neo4j** is running and update the connection details in `config.py`.
- Adjust LLM settings in `settings.py` for optimized retrieval.

## Contributing
Feel free to submit issues or pull requests to improve LexGraph AI.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
