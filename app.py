# Import required libraries
import streamlit as st
import json
import os
import pandas as pd
import chromadb
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain


# Load environment variables from .env file
load_dotenv()

def set_openai_api_key():
    """Set OpenAI API key as an environment variable."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in the .env file.")
    os.environ['OPENAI_API_KEY'] = api_key

def extract_columns_to_lists_from_dataframe(file_path, *columns):
    """
    Extracts the specified columns from an Excel file's DataFrame and outputs the values 
    from each column into individual lists.
    """
    df = pd.read_excel(file_path)
    column_data = {column: df[column].dropna().tolist() for column in columns}
    return column_data

def store_in_chromadb(columns_data):
    """Store extracted data into ChromaDB."""
    client = chromadb.Client()
    # client.delete_collection("pdpr_demo")
    collection = client.get_or_create_collection("pdpr_demo")
    answer_length = len(columns_data['Solution'])
    ids_list = [str(i) for i in range(answer_length)]
    
    documents_list = [str(problem) + " - " + str(solution) + " - " + str(reference) for problem, solution, reference in zip(columns_data["Problem"], columns_data["Solution"], columns_data["Reference"])]
    collection.add(documents=documents_list, ids=ids_list)
    return collection

def query_from_chromadb(collection, query_texts, n_results=3):
    """Query from ChromaDB."""
    results = collection.query(query_texts=query_texts, n_results=n_results)
    return results['documents'][0]

def get_response_from_langchain(context, question, model_name):
    """Query LLMChain with context and question."""
    template = """
    Given the context provided below, which is structured in the form of problem - solution - reference, please answer the following question.

    If the context directly addresses the question, provide the most relevant solution. Then, list all associated references that 
    support your solution. If the context doesn't, use your extensive knowledge in pedagogical design patterns in education to suggest 
    a potential solution and provide multiple references or sources for that insight if possible.

    Ensure the response follows this format: "Solution: [Your solution here]. References: [Your reference 1 here], [Your reference 2 here], ..."

    ---

    Context:
    {context}

    Question:
    {question}

    ---
    """


    
    prompt = PromptTemplate(template=template, input_variables=["question","context"])
    llm = OpenAI(model_name=model_name)
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    response = llm_chain.run({"question":question, "context":context})
    return response

def parse_solution_and_multiple_references(output_text):
    """
    Parse the given output text to extract the solution and its associated references.

    Args:
    - output_text (str): The generated output from the LLM.

    Returns:
    - dict: A dictionary containing the extracted solution and references.
    """
    solution_prefix = "Solution: "
    references_prefix = "References: "

    # Extract solution
    solution_start_idx = output_text.find(solution_prefix) + len(solution_prefix)
    solution_end_idx = output_text.find(references_prefix)
    solution = output_text[solution_start_idx:solution_end_idx].strip()

    # Extract references
    references_start_idx = solution_end_idx + len(references_prefix)
    references_text = output_text[references_start_idx:].strip()
    references = [ref.strip() for ref in references_text.split(",")]

    return {"Solution": solution, "References": references}

## Main Streamlit application
def main():
    st.set_page_config(
        page_title="PDPR Interface",
        page_icon="🚀",  # Change this to any emoji or image URL you want
        layout="wide",  # Using a wide layout
        initial_sidebar_state="collapsed"
    )

    # Designing the interface
    st.image("./data/logo.jpg", width=200)  # Add your logo if you have one
    st.title("PDPR Query Interface 🌐")
    st.write("Welcome to the PDPR query platform. Input your question below and get insights!")

    # Use the predefined file path
    file_path = "./data/pedagology dataset creation updated.xlsx"
    columns_data = extract_columns_to_lists_from_dataframe(file_path, "Problem", "Solution", "Reference")

    # Store data in ChromaDB
    collection = store_in_chromadb(columns_data)

    # Text input for user to enter the question
    with st.container():
        st.subheader("Your Question:")
        question_p = st.text_area("")
        
        if st.button("Query 🚀"):
            with st.spinner("Fetching response..."):
                # Query from ChromaDB
                context_extracted = query_from_chromadb(collection, [question_p])

                # Get response from Langchain
                model_name = 'gpt-3.5-turbo-0613'
                response = get_response_from_langchain(context_extracted, question_p, model_name)

                parsed_output = parse_solution_and_multiple_references(response)

                # Display on Streamlit
                st.subheader("Solution:")
                st.success(parsed_output["Solution"])

                st.subheader("References:")
                for ref in parsed_output["References"]:
                    st.success(ref)

# Execute the main function if the script is run as a Streamlit app
if __name__ == "__main__":
    main()
