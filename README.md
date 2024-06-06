# Text-analysis-using-TF-IDF-metrics
This Python program processes and analyzes a collection of US Inaugural Addresses text documents. It provides functionality for querying and retrieving documents based on term frequency-inverse document frequency (TF-IDF) metrics.

Overview:

The program utilizes natural language processing techniques to preprocess and analyze text data from a corpus of US Inaugural Addresses. It implements tokenization, stop word removal, stemming, and TF-IDF calculations to enable efficient text querying and retrieval.

Features
- Document Processing: Read and tokenize text documents, filter stopwords, and apply stemming.
- TF-IDF Calculation: Calculate term frequency (TF) and inverse document frequency (IDF) to determine term weights.
- Querying: Perform queries to find the most similar document in the corpus based on cosine similarity.
- Text Preprocessing: Convert text to lowercase, remove punctuation, and stem tokens for consistency.

Usage

Data Preparation:

Ensure the corpus of US Inaugural Addresses is available in the specified directory (./US_Inaugural_Addresses).

The corpus should contain text files named according to their respective addresses.

Environment Setup:

Install the required dependencies by running pip install -r requirements.txt.

Ensure Python 3.x environment is set up.

Run the Program:

1. Execute the main script to perform text analysis and querying.
2. Adjust the query strings or modify the code as needed for specific use cases.

Interpret Results:

- Analyze IDF values, token weights, and query results printed by the program.
- Use the insights to understand document similarities and retrieve relevant information.

Requirements:
- Python 3.x
- nltk (Natural Language Toolkit)
- numpy
- collections
- os
- math
