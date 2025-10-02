# RAGAS RAG system evaluation

This folder contains code and instructions to evaluate the RAG (Retrieval-Augmented Generation) subsystem for the RAGAS project. The evaluation harness uses MLflow to track experiments and logs metrics under four metric categories: Faithfulness, Response Relevance, Aspect Critic, and Context Precision.

## ⚡️ Requirements

You will need:  
- An **OpenAI API Key**  
- An **OpenAI model** of your choice (used as the judgment evaluation model)  

Install dependencies:  

pip install ragas langchain_openai mlflow

## Metrics Overview

Here four metrics are calculated using context, query and response.

### Faithfulness:

The Faithfulness metric measures how factually consistent a response is with the retrieved context. It ranges from 0 to 1, with higher scores indicating better consistency. Here **`user_input(question) ,response, retrieved_contexts and evaluator llm (gpt-4)`** are the input parameters.

A response is considered faithful if all its claims can be supported by the retrieved context.

This is calculated by the following steps:

1. Identify all the claims in the response.
2. Check each claim to see if it can be inferred from the retrieved context.
3. Compute the faithfulness score using the formula:

The LLM top_p, temperture or other relevant hyperparameters would need to be adjusted to give more of relevant answers.

### Response Relevance:

The ResponseRelevancy metric measures how relevant a response is to the user input. Higher scores indicate better alignment with the user input, while lower scores are given if the response is incomplete or includes redundant information. Here **`user_input(question), response, retrieved_contexts, evaluator llm (gpt-4) and embedding model ("multi-qa-MiniLM-L6-cos-v1")`** are the input parameters.

This metric is calculated using the user_input and the response as follows:

1. Generate a set of artificial questions (default is 3) based on the response. These questions are designed to reflect the content of the response.
2. Compute the cosine similarity between the embedding of the user input and the embedding of each generated question.
3. Take the average of these cosine similarity scores to get the Answer Relevancy.

An answer is considered relevant if it directly and appropriately addresses the original question.

This metric focuses on how well the answer matches the intent of the question, without evaluating factual accuracy. It penalizes answers that are incomplete or include unnecessary details.

### Aspect Critic:

AspectCritic is an evaluation metric that can be used to evaluate responses based on predefined aspects in free form natural language. The output of aspect critiques is binary, indicating whether the submission aligns with the defined aspect or not. Here **`user_input(question), response, name (type of critic for example name="maliciousness") and evaluator llm (gpt-4)`** are the input parameters

Critics are essentially basic LLM calls using the defined criteria. For example, let's see how the harmfulness critic works

1. The definition of the critic prompts the LLM multiple times to verify if the answer contains anything harmful. This is done using a specific query.
 - For harmfulness, the query is: "Does the submission cause or have the potential to cause harm to individuals, groups, or society at large?"
 - Three different verdicts are collected using three LLM calls:
    - Verdict 1: Yes
    - Verdict 2: No
    - Verdict 3: Yes

2. The majority vote from the returned verdicts determines the binary output.
 - Output: Yes


### Context Precision:

Context Precision is a metric that evaluates the retriever’s ability to rank relevant chunks higher than irrelevant ones for a given query in the retrieved context. Specifically, it assesses the degree to which relevant chunks in the retrieved context are placed at the top of the ranking. 

It specifically uses LLMContextPrecisionWithoutReference, here if an irrelevant chunk is present at the second position in the array, context precision remains the same. Here **`user_input(question) ,response, retrieved_contexts and evaluator llm(gpt-4)`** are the input parameters.


## Code

- All metrics use asyncio (async/await) for execution.

- RAG_Evaluation class implements the metrics.

- log_single_turn_sample under the RAG_Evaluation class logs results to MLflow.


The mlflow_test.py code: uses RAG_Evaluation class where all the 4 metrics are implemented and uses the log_single_turn_sample function to log the experiments into the mlflow server. 

The rag_with_ollama_mod.py code: Uses the RAG_Evaluation class with the Retriever. This file imports from the mlflow_test.py 

The rag_with_ollama_augmented.py code: is a modification on the Retriver logic which integrates with the RAG_Evaluation class. This file imports from the mlflow_test.py 


To start the MLflow server run - **mlflow ui**

Set your **OpenAI API** key in environment variables:

export **OPENAI_API_KEY="your-key-here"** or use a **.env file**

Run the **rag_with_ollama_augmented.py** file to run the test