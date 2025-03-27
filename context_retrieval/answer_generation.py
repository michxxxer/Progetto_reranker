import voyageai
import tomllib
import numpy as np
import pandas as pd
import time
import os
import csv
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_voyageai import VoyageAIEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from elasticsearch import Elasticsearch


# Load config and initialize necessary components
def load_toml() -> dict:
    with open("../config.toml", "rb") as f:
        return tomllib.load(f)


config = load_toml()
load_dotenv()


def context_extract_list(documenti: list):
    context = []
    for indice in documenti:
        context.append(indice.page_content)
    return context


def context_extract_list_ranked(documenti: list):
    context = []
    for indice in documenti:
        context.append(indice.document)
    return context


def context_extract(documenti: list):
    return ".".join(documenti)


def ranking(query: str, documents: list, model: str, top_k: int) -> list:
    vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
    reranked_results = vo.rerank(query, documents, model, top_k)
    return reranked_results


llm = ChatOpenAI(
    model="o3-mini",
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

ranked = config["embedding"]["ranked"]

elastic = Elasticsearch(
    hosts=[f"http://{config['elastic']['url']}"],
    basic_auth=(config["elastic"]["user"], config["elastic"]["password"]),
    max_retries=10
)


def get_context(question: str) -> list:
    embedding = config["embedding"]["model"]
    if (embedding.startswith("text-embedding")):
        embedding = OpenAIEmbeddings(model=config["embedding"]["model"])
    else:
        embedding = VoyageAIEmbeddings(model=config["embedding"]["model"])

    vector_store = ElasticsearchStore(es_connection=elastic,
                                      index_name=config["upload"]["index_name"] + "_" + config["embedding"]["model"],
                                      strategy=ElasticsearchStore.ApproxRetrievalStrategy(),
                                      embedding=embedding
                                      )

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    retrieved_context_list = retriever.invoke(question)
    retrieved_context = context_extract_list(retrieved_context_list)
    return retrieved_context


def get_context_ranked(question: str, context: list) -> list:
    context_ranked = ranking(question, context, config["embedding"]["ranker"], 3)
    # context_ranked = context_extract_list_ranked(context_ranked.results)
    return context_ranked

def count_tokens(context_ranked: list) -> int:
    return context_ranked.total_tokens


def list_context_position(context: list, context_ranked: list) -> list:
    result = []
    for string in context_ranked:
        try:
            position = context.index(string)
            result.append(position)
        except ValueError:
            result.append(-1)
    return result


def get_answer(question: str, context: str):
    try:
        with open("../prompts/prompt.txt", "r") as f:
            template = f.read()

        prompt = PromptTemplate.from_template(
            template=template,
            partial_variables={"context": context}
        )
        chain = (prompt | llm)
        response = chain.invoke(question)
        return response
    except Exception as e:
        print(f"Error in getting answer: {e}")
        return "Error generating answer", {"total_tokens": 0}


def process_csv_file(input_file, output_file):
    # Read the input CSV
    df = pd.read_csv(input_file)

    # Create output dataframe with required columns
    output_data = []

    # Process each row
    for index, row in df.iterrows():
        question = row['question']
        original_context = row['original_context']
        original_answer = row['original_answer']

        print(f"Processing question {index + 1}/{len(df)}: {question[:50]}...")


        # Get contexts from retrieval
        retrieved_contexts = get_context(question)
        tokens = 0
        ranked_position = []

        # Rank contexts if configured
        if ranked:
            ranked_contexts = get_context_ranked(question, retrieved_contexts)
            tokens_ranker=count_tokens(ranked_contexts)
            retrieved_contexts_list = context_extract_list_ranked(ranked_contexts.results)
            ranked_position = list_context_position(retrieved_contexts, retrieved_contexts_list)
            retrieved_context_combined = context_extract(retrieved_contexts_list)
        else:
            retrieved_contexts_list = retrieved_contexts[:3]
            retrieved_context_combined = context_extract(retrieved_contexts_list)
            tokens_ranker = None

            # Get answer based on retrieved context
        answer = get_answer(question, retrieved_context_combined)
        tokens_input = answer.usage_metadata['input_tokens']
        tokens_output = answer.usage_metadata['output_tokens']
        answer = answer.content

        # Add to output data
        output_data.append({
                'question': question,
                'original_context': original_context,
                'original_answer': original_answer,
                'retrieved_context': retrieved_contexts_list,
                'answer': answer,
                'token_used_reranking': tokens_ranker,
                'token_used_input': tokens_input,
                'token_used_output': tokens_output,
                'ranked_position': ranked_position
            })

        # Write partial results to avoid losing progress
        temp_df = pd.DataFrame(output_data)
        temp_df.to_csv(output_file, index=False)

    # Create final dataframe and write to CSV
    result_df = pd.DataFrame(output_data)
    result_df.to_csv(output_file, index=False)
    print(f"Processing complete. Results saved to {output_file}")

    return len(result_df)


def main():
    input_file = "questions.csv"
    output_file = "500answers_UR_small.csv"

    print(f"Processing questions from {input_file}")
    print(f"Results will be saved to {output_file}")

    num_questions = process_csv_file(input_file, output_file)

    print(f"- Questions processed: {num_questions}")


if __name__ == "__main__":
    main()