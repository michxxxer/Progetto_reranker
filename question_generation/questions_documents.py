import voyageai
import tomllib
import numpy as np
import time
import os
import pandas as pd
import re
from elasticsearch2.document_loader import DocumentLoader
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_voyageai import VoyageAIEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from elasticsearch import Elasticsearch
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_toml() -> dict:
    with open("../config.toml", "rb") as f:
        return tomllib.load(f)


def context_extract(documenti: list):
    context = ""
    for indice in documenti:
        context += " " + indice.page_content
    return context


def parse_questions(response_text):
    """Parse questions and answers from the LLM response."""
    questions = []
    # Regular expression to match question and answer patterns
    pattern = r"QUESTION \d+: (.*?)\n- Answer: (.*?)(?:\n- Context: (.*?))?(?:\n\n|$)"
    matches = re.findall(pattern, response_text, re.DOTALL)

    for match in matches:
        question = match[0].strip()
        answer = match[1].strip()
        # Extract context if available, otherwise use empty string
        context = match[2].strip() if len(match) > 2 and match[2] else ""

        questions.append({
            "question": question,
            "original_answer": answer,
            "source_context": context
        })

    return questions


def generate_questions_for_document(document_text, document_index=0, num_questions=3):
    """Generate relevant questions from a document using an LLM."""
    llm = ChatOpenAI(
        model="o3-mini",
        max_tokens=None,
        timeout=None
    )

    # Truncate document if too long
    max_chars = 20000
    if len(document_text) > max_chars:
        document_text = document_text[:max_chars] + "..."

    prompt = PromptTemplate.from_template(template=open("../prompts/question_generation_prompt.txt").read(),
                                          partial_variables={"num_questions": num_questions}
                                          )
    chain = (prompt | llm)

    response = chain.invoke(document_text)

    # Parse the questions from the response
    questions = parse_questions(response.content)

    return questions


def process_document_collection(documents: list, output_csv="questions.csv"):
    all_questions = []

    # Check if CSV file already exists and load existing questions
    existing_questions = []
    if os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv)
            existing_questions = existing_df.to_dict('records')
            print(f"Found existing CSV with {len(existing_questions)} questions.")
        except Exception as e:
            print(f"Error reading existing CSV: {e}")

        # Start with existing questions
        all_questions.extend(existing_questions)

        for i, doc in enumerate(documents):
            print(f"Processing document {i + 1}/{len(documents)}")
            doc_questions = generate_questions_for_document(doc, i)
            all_questions.extend(doc_questions)

            # Create and update the CSV file after each document
            df = pd.DataFrame(all_questions)
            # If file exists, append without header; otherwise, write with header
            df.to_csv(output_csv, index=False)
            print(f"Added {len(doc_questions)} questions. CSV updated with {len(all_questions)} total questions.")

            # Avoid rate limiting if needed
            # time.sleep(1)

        return all_questions

    return all_questions


def main():
    config = load_toml()
    load_dotenv()

    # Load and process the document
    documento = DocumentLoader().load_and_split_file("./libro/Quattordicesimo_capitolo.pdf")
    documento = context_extract(documento)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=11500, chunk_overlap=600, add_start_index=True
    )
    documenti = text_splitter.split_text(documento)
    documenti.pop()
    # Generate questions and save to CSV
    questions = process_document_collection(documenti, output_csv="questions.csv")

    print(f"Generation complete. Total questions generated: {len(questions)}")
    print(f"Questions saved to questions.csv")


if __name__ == "__main__":
    main()