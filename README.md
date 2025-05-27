# AGENTE VLSI–OpenROAD Assistant

A knowledge-based assistant that combines retrieval-augmented generation (RAG) with OpenROAD's Python API.  
Ask VLSI/OpenROAD questions, automatically run the generated Python in the OpenROAD binary, and get final answers—all in one end-to-end pipeline.

## Features

- Processes VLSI documentation and tutorials
- Combines graph-based and vector-based knowledge retrieval
- Specializes in VLSI design automation questions
- Generates Python/Tcl scripts for OpenROAD automation
- Provides technical explanations for VLSI concepts

## Prerequisites

- Python 3.8+
- OpenROAD CLI installed and on your `$PATH` (verify with `openroad -version`)
- A running Neo4j instance (optional, for KG lookups)
- Docker (optional, for building/running the Docker agent)

## Installation

1. Clone this repo:
   ```bash
   git clone <repo-url>
   cd graphrag-vlsi
   ```

2. (Optional) Build & run the Docker-agent:
   ```bash
   docker build -t camel-agent .
   docker run -it camel-agent
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root (this file is git-ignored):
   ```ini
   # OpenAI
   OPENAI_API_KEY=sk-<your-key>
   
   # Neo4j (optional – skip if not using the KG)
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=secret
   ```

## Project Structure

```
graphrag-vlsi/
├── pipeline.py      # sets up embeddings, vector store, KG agent & ChatAgent
├── executor.py      # CLI + helpers to extract/run code under openroad -python
├── .env             # your environment variables
└── README.md        # this file
```

## Usage

### 1) One-shot CLI

Ask a question, get a response, and run the code in OpenROAD.

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

BSD 2-Clause "Simplified" License

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
            