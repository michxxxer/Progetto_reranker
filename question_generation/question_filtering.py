import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import argparse
import time
from dotenv import load_dotenv


def check_answerability(question, context, llm):
    """
    Check if a question can be answered given the context.

    Args:
        question (str): The question to check
        context (str): The context to use for answering
        llm: The language model to use

    Returns:
        bool: Whether the question is answerable
    """
    prompt = PromptTemplate.from_template("""
You are an expert at determining whether a question can be answered based on provided context.

Context:
```
{context}
```

Question: {question}

Evaluate if the question can be answered using ONLY the information in the context.
Consider:
1. If the answer is explicitly stated in the context
2. If the answer can be reasonably inferred from the context
3. If the context lacks the necessary information to answer the question

First, think step by step about what information is needed to answer the question.
Then, determine if the question can be answered with the given context.

Return ONLY a JSON object with this format:
{{
    "answerable": true/false
}}
""")

    chain = (prompt | llm)

    try:
        response = chain.invoke({"question": question, "context": context})
        # Extract JSON from response
        import json
        import re

        json_match = re.search(r'({.*})', response.content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(1))
            return result.get("answerable", False)
        else:
            print(f"Warning: Failed to parse response for question: {question[:50]}...")
            return False
    except Exception as e:
        print(f"Error processing question: {e}")
        return False


def process_csv(input_file, output_file, remove_unanswerable=True, model="gpt-4o-mini", api_key=None, batch_size=10):
    """
    Process a CSV file and check if questions can be answered with the given context.

    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
        remove_unanswerable (bool): Whether to remove unanswerable questions
        model (str): The model to use
        api_key (str): OpenAI API key
        batch_size (int): Number of questions to process before saving
    """
    # Load the CSV file
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} questions from {input_file}")

    # Initialize LLM
    llm = ChatOpenAI(
        model="o3-mini",
        max_tokens=None,
        timeout=None
    )

    # Add column for results
    df["answerable"] = False

    # Process questions
    processed_count = 0

    for i, row in df.iterrows():
        question = row["question"]
        context = row["original_context"]
        answer = row["original_answer"]

        print(f"Processing question {i + 1}/{len(df)}: {question[:50]}...")

        # Check if question can be answered
        answerable = check_answerability(question, context, llm)

        # Update DataFrame
        df.at[i, "answerable"] = answerable

        processed_count += 1

        # Save intermediate results
        if processed_count % batch_size == 0:
            temp_df = df.copy()
            if remove_unanswerable:
                temp_df = temp_df[temp_df["answerable"]]
            temp_df.to_csv(output_file, index=False)
            print(f"Saved intermediate results ({processed_count}/{len(df)} processed)")

        # Add a small delay to avoid rate limiting
        time.sleep(0.5)

    # Save final results
    final_df = df.copy()
    if remove_unanswerable:
        final_df = final_df[final_df["answerable"]]
        print(f"Removed {len(df) - len(final_df)} unanswerable questions")

    final_df.to_csv(output_file, index=False)
    print(f"Saved {len(final_df)} questions to {output_file}")

    return final_df


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Check if questions can be answered with the given context")
    parser.add_argument("--input", "-i", type=str, default="questions.csv", help="Input CSV file")
    parser.add_argument("--output", "-o", type=str, default="answerable_questions.csv", help="Output CSV file")
    parser.add_argument("--keep-all", "-k", action="store_true", help="Keep all questions including unanswerable ones")
    parser.add_argument("--model", "-m", type=str, default="o3-mini", help="OpenAI model to use")
    parser.add_argument("--batch-size", "-b", type=int, default=10, help="Batch size for saving intermediate results")

    args = parser.parse_args()

    process_csv(
        input_file=args.input,
        output_file=args.output,
        remove_unanswerable=not args.keep_all,
        model=args.model,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()