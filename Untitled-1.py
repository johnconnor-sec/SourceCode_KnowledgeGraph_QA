
import argparse
import os

from dotenv import load_dotenv
from langchain.chains import GraphCypherQAChain
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

# Set up OpenAI credentials
load_dotenv()

# Set up Neo4j credentials
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "password"

# Initialize Neo4j graph
graph = Neo4jGraph()

# %%
def load_source_code_to_graph(directory_path):                                                                                                                        
    try:                                                                                                                                                              
        # Load source code                                                                                                                                            
        loader = GenericLoader.from_filesystem(                                                                                                                       
            directory_path,                                                                                                                                           
            glob="*",                                                                                                                                                 
            suffixes=[".py", ".js", ".go", ".md", ".html"],  # Ensure .go is included                                                                                 
            parser=LanguageParser()                                                                                                                                   
        )                                                                                                                                                             
        documents = loader.load()                                                                                                                                     
        print(f"Loaded documents: {[doc.metadata.get('name', 'Unknown') for doc in documents]}")                                                                      
                                                                                                                                                                       
        # Split documents if they're too large                                                                                                                        
        text_splitter = RecursiveCharacterTextSplitter.from_language(                                                                                                 
            language=[Language.PYTHON,                                                                                                                                
                      Language.JS,                                                                                                                                    
                      Language.GO,  # Ensure Language.GO is defined                                                                                                   
                      Language.HTML,                                                                                                                                  
                      Language.MARKDOWN],                                                                                                                             
            chunk_size=1000,                                                                                                                                          
            chunk_overlap=0                                                                                                                                           
        )                                                                                                                                                             
        split_docs = text_splitter.split_documents(documents)                                                                                                         
        print(f"Split documents: {len(split_docs)} chunks")                                                                                                           
                                                                                                                                                                       
        # Populate Neo4j graph                                                                                                                                        
        for doc in split_docs:                                                                                                                                        
            query = """                                                                                                                                               
            MERGE (f:CodeChunk {name: $name})                                                                                                                         
            SET f.content = $content, f.language = $language                                                                                                          
            """                                                                                                                                                       
            graph.query(query, {                                                                                                                                      
                "name": doc.metadata.get('name', 'Unknown'),                                                                                                          
                "content": doc.page_content,                                                                                                                          
                "language": doc.metadata.get('language', 'Unknown')                                                                                                   
            })                                                                                                                                                        
                                                                                                                                                                       
        # Create relationships between code chunks                                                                                                                    
        graph.query("""                                                                                                                                               
        MATCH (a:CodeChunk), (b:CodeChunk)                                                                                                                            
        WHERE a <> b AND a.language = b.language                                                                                                                      
        MERGE (a)-[:SAME_LANGUAGE]->(b)                                                                                                                               
        """)                                                                                                                                                          
                                                                                                                                                                       
        # Refresh graph schema                                                                                                                                        
        graph.refresh_schema()                                                                                                                                        
        print("Graph schema updated:", graph.schema)                                                                                                                  
    except Exception as e:                                                                                                                                            
        print(f"Error loading source code: {str(e)}")                                                                                                                 
        raise  # Re-raise the exception for further debugging
                                                                                                                                                                       
 # Example usage                                                                                                                                                       
directory_path = "/home/sarah/Documents/AI_Code/fabric"                                                                                                                             
load_source_code_to_graph(directory_path) 

# %%
def setup_qa_chain():
    try:
        # Ensure we have a valid graph schema
        graph.refresh_schema()
        if not graph.schema:
            raise ValueError("Graph schema is empty. Make sure data has been loaded into the graph.")

        # Initialize the language model
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        # Create the GraphCypherQAChain
        chain = GraphCypherQAChain.from_llm(
            graph=graph,
            llm=llm,
            verbose=True,
            validate_cypher=True,
            graph_schema=graph.schema,  # Explicitly provide the graph schema
            cypher_query_corrector=None,  # Optional: You can implement a custom corrector if needed
            return_intermediate_steps=True,  # This will return the Cypher query used
            return_direct=False  # Set to True if you want to return the raw graph query results
        )
        
        print("QA Chain setup successfully.")
        return chain
    except Exception as e:
        print(f"Error setting up QA Chain: {str(e)}")
        return None
    else:
        return chain

# %%
def answer_query(chain, query):
    response = chain.invoke({"query": query})
    return response['result']

# %%
def main():
    # Get the directory path from user input (for Jupyter Notebook)
    # directory_path = input("Enter the path to the directory containing the source code: ")

    # Load source code into graph
    print(f"Loading source code from: {directory_path}")
    load_source_code_to_graph(directory_path)

    # Set up QA chain
    qa_chain = setup_qa_chain()

    # Interactive Q&A loop
    while True:
        user_query = input("Ask a question about the code (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        answer = answer_query(qa_chain, user_query)
        print("Answer:", answer)
        print()

# Run the main function
main()


