import pandas as pd
import ast
import csv
import numpy as np
from deepeval import evaluate
from deepeval.metrics import ContextualRecallMetric
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.metrics.ragas import RAGASContextualRecallMetric
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv
import os
import time
from tqdm import tqdm


def calculate_metrics(input_file, output_file):
    # Load environment variables
    load_dotenv()

    # Read the input CSV file
    df = pd.read_csv(input_file)

    # Create empty columns for results
    df['ContextRelevancy'] = None
    df['ContextRelevancyAVG'] = None
    df['ContextRecall'] = None
    df['ContextRecallAVG'] = None
    df['RagasContextRecall'] = None
    df['RagasContextRecallAVG'] = None
    df['ContextualPrecision'] = None
    df['ContextualPrecisionAVG'] = None

    # Process each row in the dataframe
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        question = row['question']
        original_context = row['original_context']
        original_answer = row['original_answer']
        retrieved_context = row['retrieved_context']
        answer = row['answer']

        # Create context lists
        context = [original_context]
        retrieval_context = ast.literal_eval(retrieved_context)
        print(context[0])
        # Create test case
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=original_answer,
            context=context,
            retrieval_context=retrieval_context
        )

        # Calculate ContextualRelevancy metric
        context_relevancy_scores = []
        for _ in range(1):
            try:
                metric = ContextualRelevancyMetric(
                    threshold=0.7,
                    model="gpt-4o-mini"
                )
                metric.measure(test_case)
                context_relevancy_scores.append(metric.score)
            except Exception as e:
                print(f"Error calculating ContextRelevancy for question {index}: {str(e)}")
                continue

        # Calculate average ContextRelevancy if we have valid scores
        if context_relevancy_scores:
            df.at[index, 'ContextRelevancy'] = str(context_relevancy_scores)
            df.at[index, 'ContextRelevancyAVG'] = np.mean(context_relevancy_scores)

        # Calculate ContextualRecall metric
        context_recall_scores = []
        for _ in range(5):
            try:
                metric = ContextualRecallMetric(
                    threshold=0.7,
                    model="gpt-4o-mini"
                )
                metric.measure(test_case)
                context_recall_scores.append(metric.score)
            except Exception as e:
                print(f"Error calculating ContextRecall for question {index}: {str(e)}")
                continue

        # Calculate average ContextRecall if we have valid scores
        if context_recall_scores:
            df.at[index, 'ContextRecall'] = str(context_recall_scores)
            df.at[index, 'ContextRecallAVG'] = np.mean(context_recall_scores)

        # Calculate RAGASContextualRecall metric
        ragas_context_recall_scores = []
        for _ in range(5):
            try:
                metric = RAGASContextualRecallMetric(
                    threshold=0.7,
                    model="gpt-4o-mini"
                )
                metric.measure(test_case)
                ragas_context_recall_scores.append(metric.score)
            except Exception as e:
                print(f"Error calculating RAGASContextRecall for question {index}: {str(e)}")
                continue

        # Calculate average RAGASContextRecall if we have valid scores
        if ragas_context_recall_scores:
            df.at[index, 'RagasContextRecall'] = str(ragas_context_recall_scores)
            df.at[index, 'RagasContextRecallAVG'] = np.mean(ragas_context_recall_scores)

        """
        # Calculate ContextualPrecision metric - commented out as in original
        context_precision_scores = []
        for _ in range(3):
            try:
                metric = ContextualPrecisionMetric(
                    threshold=0.7,
                    model="gpt-4o-mini"
                )
                metric.measure(test_case)
                context_precision_scores.append(metric.score)
            except Exception as e:
                print(f"Error calculating ContextualPrecision for question {index}: {str(e)}")
                continue

        # Calculate average ContextualPrecision if we have valid scores
        if context_precision_scores:
            df.at[index, 'ContextualPrecision'] = str(context_precision_scores)
            df.at[index, 'ContextualPrecisionAVG'] = np.mean(context_precision_scores)
        """

    # Select only the relevant columns for output
    result_df = df[['question', 'ContextRelevancy', 'ContextRelevancyAVG',
                    'ContextRecall', 'ContextRecallAVG',
                    'RagasContextRecall', 'RagasContextRecallAVG',
                    'ContextualPrecision', 'ContextualPrecisionAVG']]

    # Write the results to CSV
    result_df.to_csv(output_file, index=False)

    print(f"Metrics calculation complete. Results saved to {output_file}")


if __name__ == "__main__":
    input_file = "500answers_URcopy.csv"  # Default input file
    output_file = "500evaluation_UR.csv"  # Output file

    calculate_metrics(input_file, output_file)