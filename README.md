# GraphRAG VLSI Assistant

A knowledge-based assistant that combines graph-based and retrieval-augmented generation (RAG) approaches to answer VLSI design automation questions. The system uses Neo4j for graph storage and Qdrant for vector storage, powered by OpenAI's language models.

## Features

- Processes VLSI documentation and tutorials
- Combines graph-based and vector-based knowledge retrieval
- Specializes in VLSI design automation questions
- Generates Python/Tcl scripts for OpenROAD automation
- Provides technical explanations for VLSI concepts

## Prerequisites

- Python 3.8+
- OpenAI API key
- Neo4j database access
- Sufficient disk space for vector storage

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd graphrag-vlsi
```

2. Install required dependencies:
```bash
pip install python-dotenv camel-ai colorama neo4j qdrant-client
```

3. Create a `.env` file in the project root with your API keys:
```env
OPENAI_API_KEY=your_openai_api_key_here
# Optional: Override default Neo4j credentials
# NEO4J_URI=your_neo4j_uri
# NEO4J_USERNAME=your_username
# NEO4J_PASSWORD=your_password
```

## Project Structure

```
graphrag-vlsi/
├── main.py              # Main script
├── flow_tutorial.md     # VLSI tutorial/documentation
├── .env                 # Environment variables
├── my_vectors/         # Vector storage directory
└── README.md           # This file
```

## Usage

1. Prepare your VLSI documentation:
   - Place your VLSI tutorial or documentation in `flow_tutorial.md`
   - The file should contain relevant information about VLSI design flows, techniques, and best practices

2. Run the assistant:
```bash
python main.py
```

The script will:
- Initialize the knowledge bases
- Process the documentation
- Execute a predefined query about IR drop analysis
- Generate a detailed response with relevant code

## Default Query

The system comes with a default query:
```python
"how can i perform IR drop analysis on the M1 layer where the pins are located, provide python code"
```

To modify the query, edit the `query` variable in `main.py`.

## Components

- **Neo4j Graph**: Stores relationships between VLSI concepts
- **Qdrant Vector Store**: Maintains embeddings for semantic search
- **CAMEL Agents**: 
  - Knowledge Graph Agent: Extracts structured information
  - Chat Agent: Generates responses using OpenAI models
- **Vector Retriever**: Performs similarity-based document retrieval

## Troubleshooting

1. **Qdrant Lock Issues**
   If you encounter Qdrant lock errors, run:
   ```bash
   ps aux | grep qdrant
   kill -9 <PID>
   ```

2. **Neo4j Connection Issues**
   - Verify your Neo4j credentials in `.env`
   - Ensure the Neo4j database is running and accessible

3. **OpenAI API Issues**
   - Check your API key in `.env`
   - Verify you have sufficient API credits

## Contributing

Feel free to submit issues and enhancement requests!

## License

[Your chosen license]

## Acknowledgments

- Built with [CAMEL-AI](https://github.com/camel-ai/camel)
- Uses OpenAI's language models
- Powered by Neo4j and Qdrant

**files**

- `docker_agent.py` - the main file for the docker agent
- `requirements.txt` - the requirements for the docker agent
- `Dockerfile` - the Dockerfile for the docker agent
- `flow_tutorial.md` - the flow tutorial for the docker agent

**run**

```bash
docker build -t camel-agent .
docker run -it camel-agent
```
            
