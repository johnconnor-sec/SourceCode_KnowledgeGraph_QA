import argparse
import os

from dotenv import load_dotenv
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
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
            glob="**/*",
            suffixes=[".py", ".js", ".go", ".md", ".html"],
            exclude=[r"^\..*", r"README.md", r"system.md", r"LICENSE", r"go.sum", r"system.md", r"patterns/**/*"],  # Exclude .dot files and virtual environments
            parser=LanguageParser()
        )
        documents = loader.load()
        
        # Explicitly set document names
        for doc in documents:
            if 'source' in doc.metadata:
                doc.metadata['name'] = os.path.basename(doc.metadata['source'])
            else:
                doc.metadata['name'] = 'Unknown'
        
        print(f"Loaded documents: {[doc.metadata.get('name', 'Unknown') for doc in documents]}")
                                                                                                                                                                       
        # Split documents if they're too large                                                                                                                        
        languages = [                                                                                                                                                 
            Language.PYTHON,                                                                                                                                          
            Language.JS,                                                                                                                                              
            Language.GO,                                                                                                                                              
            Language.HTML,                                                                                                                                            
            Language.MARKDOWN                                                                                                                                         
        ]                                                                                                                                                             
                                                                                                                                                                      
        for lang in languages:                                                                                                                                        
            text_splitter = RecursiveCharacterTextSplitter.from_language(                                                                                             
                language=lang,                                                                                                                                        
                chunk_size=1000,                                                                                                                                      
                chunk_overlap=0                                                                                                                                       
            )                                                                                                                                                         
            split_docs = text_splitter.split_documents(documents)                                                                                                     
            print(f"Split documents for {lang}: {len(split_docs)} chunks")                                                                                            
                                                                                                                                                                      
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
        # Consider logging the error or handling specific exceptions
        raise
                                                                                                                                                                       
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
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # Changed from gpt-4o-mini to gpt-3.5-turbo

        # Create a custom prompt template
        CYPHER_GENERATION_TEMPLATE = """
        You are an AI assistant that helps to convert natural language questions about a codebase into Cypher queries.
        Use the following neo4j graph schema to generate Cypher queries:
        {schema}

        The graph contains CodeChunk nodes with properties: content, language, and name.
        CodeChunks with the same language are connected by SAME_LANGUAGE relationships.

        When searching for specific terms, use CONTAINS or =~ for partial matches.
        Always return the content of the CodeChunk nodes in your queries.
        Do not include the word 'cypher' in your query.

        Human: {question}
        AI: Based on the given schema, here's a Cypher query to answer your question:

        """

        cypher_prompt = PromptTemplate(
            input_variables=["schema", "question"],
            template=CYPHER_GENERATION_TEMPLATE
        )

        # Create the GraphCypherQAChain
        chain = GraphCypherQAChain.from_llm(
            graph=graph,
            cypher_llm=llm,
            qa_llm=llm,
            verbose=True,
            cypher_prompt=cypher_prompt,
            validate_cypher=True,
            return_intermediate_steps=True,
            return_direct=False,
            allow_dangerous_requests=True
        )

        print("QA Chain setup successfully.")
        return chain
    except Exception as e:
        print(f"Error setting up QA Chain: {str(e)}")
        raise  # Re-raise the exception instead of returning None

# %%
def answer_query(chain, query):
    try:
        response = chain.run(query)
        
        # print("Full response:", response)  # Add this line for debugging
        
        if isinstance(response, dict) and 'intermediate_steps' in response:
            cypher_query = response['intermediate_steps'][0]['query']
            print("Generated Cypher query:")
            print(cypher_query)
            
            cypher_result = response['intermediate_steps'][0]['result']
            print("Cypher query result:")
            print(cypher_result)
            
            if not cypher_result:
                return "No matching data found in the graph for this query."
            
            # Process the cypher_result to generate a meaningful answer
            if isinstance(cypher_result, list) and len(cypher_result) > 0:
                content = cypher_result[0].get('c.content', '')
                if content:
                    return f"Found relevant content: {content[:1000]}..."
                else:
                    return "No relevant content found in the matched nodes."
            else:
                return f"Unexpected result format: {cypher_result}"
        
        # If the response is a string, return it directly
        if isinstance(response, str):
            return response
        
        # If we reach here, we couldn't find a proper answer
        return f"Unable to generate an answer. Response structure: {type(response)}"
    except Exception as e:
        print(f"Error in answer_query: {str(e)}")
        return f"An error occurred while processing your query: {str(e)}"

# %%
def main():
    try:
        print(f"Loading source code from: {directory_path}")
        load_source_code_to_graph(directory_path)
        
        # Print some statistics about the loaded data
        node_count = graph.query("MATCH (n:CodeChunk) RETURN count(n) as count")[0]['count']
        print(f"Loaded {node_count} code chunks into the graph.")

        qa_chain = setup_qa_chain()

        while True:
            user_query = input("Ask a question about the code (or type 'exit' to quit): ")
            if user_query.lower() == 'exit':
                break
            try:
                answer = answer_query(qa_chain, user_query)
                print("Answer:", answer)
            except Exception as e:
                print(f"An error occurred while processing the query: {str(e)}")
            print()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    print("Exiting the program.")

# Run the main function
if __name__ == "__main__":
    main()




