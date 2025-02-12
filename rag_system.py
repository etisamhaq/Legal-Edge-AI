
from typing import Dict, TypedDict, List, Tuple, Annotated, Sequence, Literal
from langgraph.graph import StateGraph, END
import operator
from neo4j import GraphDatabase
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pydantic import BaseModel
import json

class AgentState(TypedDict):
    messages: List[dict]
    context: List[str]
    current_query: str
    retrieved_docs: List[dict]
    processed_response: str
    needs_retrieval: bool
    needs_validation: bool
    next_step: str
    attempts: int  # Add counter to prevent infinite loops

# Knowledge Base Interface
class KnowledgeBase:
    def __init__(self, uri, username, password):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def close(self):
        self.driver.close()
    
    def retrieve_documents(self, query: str, top_k: int = 3) -> List[dict]:
        with self.driver.session() as session:
            # Get articles from Neo4j
            result = session.run("""
                MATCH (a:Article)
                WHERE a.content IS NOT NULL
                RETURN ID(a) as id, a.content as content, 
                       COALESCE(a.name, a.title) as title
            """)
            articles = [(r["id"], r["content"], r["title"]) for r in result]
            
            if not articles:
                return []
            
            # Generate embeddings and find similar documents
            query_embedding = self.model.encode([query])
            article_embeddings = self.model.encode([art[1] for art in articles])
            similarities = cosine_similarity(query_embedding, article_embeddings)[0]
            
            # Get top matches
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            return [
                {
                    "id": articles[idx][0],
                    "content": articles[idx][1],
                    "title": articles[idx][2],
                    "similarity": float(similarities[idx])
                }
                for idx in top_indices
            ]

# LLM Interface
class LLMInterface:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
    
    def generate_response(self, messages: List[dict]) -> str:
        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model="mixtral-8x7b-32768",
            temperature=0.7,
            max_tokens=1000
        )
        return chat_completion.choices[0].message.content

class RAGNodes:
    def __init__(self, knowledge_base: KnowledgeBase, llm: LLMInterface):
        self.kb = knowledge_base
        self.llm = llm
        self.max_attempts = 3  # Maximum number of retrieval/validation attempts
    
    def query_router(self, state: AgentState) -> AgentState:
        """Route the query to appropriate next step"""
        # Check if we've exceeded maximum attempts
        if state["attempts"] >= self.max_attempts:
            # Force proceed to generation with whatever context we have
            state["needs_retrieval"] = False
            state["needs_validation"] = False
            state["next_step"] = "generate"
            return state

        if state["needs_retrieval"]:
            state["next_step"] = "retrieve"
        elif state["needs_validation"]:
            state["next_step"] = "validate"
        else:
            state["next_step"] = "generate"
        return state
    
    def retrieve_context(self, state: AgentState) -> AgentState:
        """Retrieve relevant documents from knowledge base"""
        state["attempts"] += 1
        docs = self.kb.retrieve_documents(state["current_query"])
        
        if not docs:  # If no documents found, proceed to generate
            state["needs_retrieval"] = False
            state["needs_validation"] = False
            state["next_step"] = "generate"
            return state
            
        state["retrieved_docs"] = docs
        state["context"] = [doc["content"] for doc in docs]
        state["needs_retrieval"] = False
        state["needs_validation"] = True
        state["next_step"] = "route"
        return state
    
    def validate_context(self, state: AgentState) -> AgentState:
        """Validate retrieved context relevance"""
        state["attempts"] += 1
        validation_prompt = {
            "role": "system",
            "content": """Analyze if the retrieved context is relevant to the query.
            Return JSON with format: {"is_relevant": boolean, "reason": string}"""
        }
        
        context_str = "\n".join(state["context"])
        user_prompt = {
            "role": "user",
            "content": f"""Query: {state["current_query"]}
            Retrieved Context: {context_str}
            Is this context relevant for answering the query?"""
        }
        
        response = self.llm.generate_response([validation_prompt, user_prompt])
        try:
            result = json.loads(response)
            state["needs_validation"] = False
            if not result["is_relevant"] and state["attempts"] < self.max_attempts:
                state["needs_retrieval"] = True
                state["context"] = []
            else:
                # Either context is relevant or we've hit max attempts
                state["needs_retrieval"] = False
                state["needs_validation"] = False
        except:
            # If validation fails, proceed with current context
            state["needs_validation"] = False
            state["needs_retrieval"] = False
        
        state["next_step"] = "route"
        return state
    
    def generate_response(self, state: AgentState) -> AgentState:
        """Generate final response using LLM"""
        context_str = "\n".join(state["context"])
        
        # Adjust prompt based on whether we have context
        if not context_str.strip():
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. If you don't have enough information, acknowledge that and provide a general response."
                },
                {
                    "role": "user",
                    "content": f"Question: {state['current_query']}\n\nPlease note that I couldn't find specific context for this query."
                }
            ]
        else:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on the provided context."
                },
                {
                    "role": "user",
                    "content": f"""Context: {context_str}
                    
                    Question: {state["current_query"]}
                    
                    Please provide a detailed answer based on the above context."""
                }
            ]
        
        response = self.llm.generate_response(messages)
        state["processed_response"] = response
        state["next_step"] = "end"
        return state

class RAGSystem:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, groq_api_key: str):
        self.knowledge_base = KnowledgeBase(neo4j_uri, neo4j_user, neo4j_password)
        self.llm = LLMInterface(groq_api_key)
        self.nodes = RAGNodes(self.knowledge_base, self.llm)
        
        workflow = StateGraph(AgentState)
        
        workflow.add_node("route", self.nodes.query_router)
        workflow.add_node("retrieve", self.nodes.retrieve_context)
        workflow.add_node("validate", self.nodes.validate_context)
        workflow.add_node("generate", self.nodes.generate_response)
        
        workflow.set_entry_point("route")
        
        workflow.add_conditional_edges(
            "route",
            lambda x: x["next_step"],
            {
                "retrieve": "retrieve",
                "validate": "validate",
                "generate": "generate"
            }
        )
        
        workflow.add_edge("retrieve", "route")
        workflow.add_edge("validate", "route")
        workflow.add_edge("generate", END)
        
        self.workflow = workflow.compile()
    
    def process_query(self, query: str) -> str:
        """Process a single query through the RAG workflow"""
        initial_state: AgentState = {
            "messages": [],
            "context": [],
            "current_query": query,
            "retrieved_docs": [],
            "processed_response": "",
            "needs_retrieval": True,
            "needs_validation": False,
            "next_step": "route",
            "attempts": 0  # Initialize attempts counter
        }
        
        final_state = self.workflow.invoke(initial_state)
        return final_state["processed_response"]
    
    def chat(self):
        """Interactive chat interface"""
        print("Welcome to the LangGraph RAG Chat! (Type 'exit' to quit)")
        print("="*50)
        
        while True:
            query = input("\nYou: ").strip()
            
            if query.lower() == 'exit':
                print("Goodbye!")
                break
            
            try:
                response = self.process_query(query)
                print("\nAssistant:", response)
                print("\n" + "="*50)
            except Exception as e:
                print(f"\nError: {str(e)}")
                print("Please try again.")
    
    def close(self):
        """Clean up resources"""
        self.knowledge_base.close()

# # Initialize the RAG system
# rag = RAGSystem(
#     neo4j_uri="bolt://localhost:7687",
#     neo4j_user="neo4j",
#     neo4j_password="12345678",
#     groq_api_key="your_api_key_here"
# )

# # Method 1: Process a single query
# question = "What are the different types of supply of goods mentioned in the regulations?"
# response = rag.process_query(question)
# print(response)

# # Method 2: Interactive chat
# rag.chat()

# # Clean up when done
# rag.close()