import os
from dotenv import load_dotenv

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType, EmbeddingModelType
from camel.configs import ChatGPTConfig
from camel.embeddings import OpenAIEmbedding
from camel.storages import QdrantStorage, Neo4jGraph
from camel.retrievers import VectorRetriever
from camel.loaders import UnstructuredIO
from camel.agents import KnowledgeGraphAgent, ChatAgent
from camel.messages import BaseMessage

# 1) Load .env
load_dotenv()

# 2) Embedding + Vector Retriever
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # make sure this is set
embedding = OpenAIEmbedding(model_type=EmbeddingModelType.TEXT_EMBEDDING_3_LARGE)
vector_store = QdrantStorage(
    vector_dim=embedding.get_output_dim(),
    path="vector_db/",
    collection_name="documents_collection",
)
vector_retriever = VectorRetriever(
    embedding_model=embedding,
    storage=vector_store,
)

# 3) Knowledge-Graph Storage (Neo4j)
# Set Neo4j instance
n4j = Neo4jGraph(
    url="neo4j+s://a77d863c.databases.neo4j.io",
    username="neo4j",
    password="f1zopPMnKXlhQAYvugcoLUr8t0s9QruIyYsY0YxBBhU"

)

# 4) Unstructured IO & KG‐Agent
uio = UnstructuredIO()

# 5) OpenAI Chat Model
chat_cfg = ChatGPTConfig(temperature=0.2).as_dict()
openai_model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O_MINI,
    model_config_dict=chat_cfg,
)

# 6) KG Agent
kg_agent = KnowledgeGraphAgent(model=openai_model)

# 7) System prompt for ChatAgent
sys_msg = BaseMessage.make_assistant_message(
    role_name="VLSI Engineer",
    content=(
        "You are a VLSI design automation expert specializing in the RTL-to-GDSII flow.\n"
        "Your job has two modes, depending on the user's request:\n\n"
        "1) Script Generation Mode\n"
        "   - Produce a single, self-contained ```python``` (or hybrid Python/Tcl) script\n"
        "     that runs from synthesis through GDSII export in OpenROAD without further editing.\n"
        "   - Include all necessary imports at the top.\n"
        "   - Wrap the entire script in one fenced code block:\n"
        "     ```python\n"
        "     # your code here\n"
        "     ```\n"
        "   - Do not emit any prose, explanations, or extra text—only the runnable script.\n\n"
        "2) VLSI Q&A Mode\n"
        "   - If the user asks a question about VLSI design, physical implementation,\n"
        "     timing, power, constraints, or OpenROAD usage, provide a concise,\n"
        "     accurate technical explanation.\n"
        "   - You may include small code snippets or Tcl/API examples to illustrate your answer,\n"
        "     but keep them minimal and relevant.\n"
        "   - Precede code examples with a brief introduction in plain text,\n"
        "     and present them in fenced blocks.\n\n"
        "Choose the appropriate mode automatically and respond accordingly."
    )
)

# 8) ChatAgent
camel_agent = ChatAgent(system_message=sys_msg, model=openai_model)


def answer_vlsi_query(
    query: str,
    top_k: int = 7,
    similarity_threshold: float = 0.2
) -> str:
    """
    1) Vector‐based retrieval
    2) KG extraction & Neo4j lookups
    3) Combine contexts and ask camel_agent
    4) Return the assistant's first message
    """
    # Vector retrieval
    retrieved = vector_retriever.query(
        query=query,
        top_k=top_k,
        similarity_threshold=similarity_threshold
    )

    # KG‐agent extraction
    el = uio.create_element_from_text(text=query, element_id="kg_query")
    ans_el = kg_agent.run(el, parse_graph_elements=True)

    # Neo4j lookups
    kg_ctx = []
    for node in ans_el.nodes:
        cypher = f"""
        MATCH (n {{id: '{node.id}'}})-[r]->(m)
        RETURN 'Node ' + n.id + ' --' + type(r) + '--> ' + m.id AS desc
        UNION
        MATCH (n)<-[r]-(m {{id: '{node.id}'}})
        RETURN 'Node ' + m.id + ' --' + type(r) + '--> ' + n.id AS desc
        """
        for rec in n4j.query(query=cypher):
            kg_ctx.append(rec["desc"])

    # Combine contexts
    context = f"{retrieved}\n" + "\n".join(kg_ctx)

    # Ask the agent
    user_msg = BaseMessage.make_user_message(
        role_name="vlsi User",
        content=f"The Original Query is: {query}\n\nRetrieved Context:\n{context}"
    )
    resp = camel_agent.step(user_msg)
    return resp.msgs[0].content 