# A6: Naive RAG vs. Contextual Retrieval

* **Student ID:** 126425
* **Assigned Chapter:** Chapter 5
* **Chapter Title:** *Embeddings*

This assignment implements and compares two retrieval-based question answering approaches on the assigned textbook chapter:

1. **Naive RAG**
2. **Contextual Retrieval**

The assigned chapter was determined from the last digit of the student ID. Since my student ID ends in **5**, the assigned chapter is **Chapter 5: Embeddings**.

The assignment includes document extraction, preprocessing, chunking, vector indexing, retrieval-based answering, QA evaluation, ROUGE-based comparison, JSON result export, and a simple chatbot web application.


## Dataset 

* **Book:** *Speech and Language Processing (3rd edition draft)*
* **Chapter Used:** Chapter 5 — *Embeddings*
* **Page Range Extracted:** printed pages 96–119
* **PDF File Used:** `ed3book_jan26.pdf`

The chapter covers the following topics:

* Lexical semantics
* Vector semantics
* Count-based embeddings
* Cosine similarity
* Word2vec
* Embedding visualization
* Semantic properties of embeddings
* Bias in embeddings
* Evaluation of vector models


### 1. Document Extraction and Cleaning

The assigned chapter text was extracted from the provided PDF using **PyMuPDF (`fitz`)**. The chapter text was then cleaned to remove:

* page numbers
* repeated section markers
* formatting artifacts
* isolated bullets
* broken hyphenations
* noisy heading-only lines

This produced a cleaned text file suitable for chunking and retrieval.

### 2. Naive RAG

The **Naive RAG** pipeline was implemented using the following steps:

* Split the cleaned chapter text into chunks using `RecursiveCharacterTextSplitter`
* Chunk size: approximately **1000 characters**
* Chunk overlap: **200 characters**
* Embed chunks using **`sentence-transformers/all-MiniLM-L6-v2`**
* Store embeddings in a **FAISS** vector index
* Retrieve top-k similar chunks for a user query
* Produce answers using an extractive fallback method based on retrieved content

### 3. Contextual Retrieval

The **Contextual Retrieval** method was implemented by enriching each chunk with a locally generated contextual sentence before embedding.

Because the OpenAI API quota was unavailable during development, contextual enrichment was implemented using a **local instruction-tuned model**:

* Model used for enrichment: **`google/flan-t5-small`**
* Each chunk was given a one-sentence contextual prefix such as:

  * *“This chunk from Embeddings discusses ...”*
* The contextualized chunk was then embedded and indexed with FAISS

This follows the same overall idea as contextual chunk enrichment, where additional context is prepended before retrieval.

### 4. QA Pair Construction

A set of **20 question-answer pairs** was manually prepared from Chapter 5.
Each QA pair contains:

* a question
* a reference (ground-truth) answer
* the answer generated using Naive RAG
* the answer generated using Contextual Retrieval

### 5. Evaluation

Both methods were evaluated using:

* **ROUGE-1**
* **ROUGE-2**
* **ROUGE-L**

The generated answers were compared against manually written reference answers.

### 6. Web Application
A simple chatbot was developed.
The chatbot:

* allows users to ask questions about Chapter 5
* uses the **Contextual Retrieval** backend
* displays the generated answer
* displays the retrieved source chunk used to support the answer


## Results

### ROUGE Comparison

| Method               |  ROUGE-1 |  ROUGE-2 |  ROUGE-L |
| -------------------- | -------: | -------: | -------: |
| Naive RAG            | 0.249967 | 0.043661 | 0.160451 |
| Contextual Retrieval | 0.231563 | 0.037733 | 0.147160 |

### Interpretation

In this implementation, **Naive RAG outperformed Contextual Retrieval** on all three ROUGE metrics.

A likely reason is that the locally generated contextual summaries were often too general and introduced additional noise into the chunk representations. Instead of always improving retrieval precision, the added contextual text sometimes reduced the discriminative value of the original chunk.

Even though Contextual Retrieval did not outperform Naive RAG in this experiment, it still demonstrated the intended design principle of enriching chunks before embedding.


## Discussion

Several practical issues were encountered during development:

* environment and dependency conflicts involving `sentence-transformers`, `huggingface_hub`, and `faiss`
* FAISS compatibility issues with NumPy
* unreliable short-form generation from FLAN-T5 for direct answer generation
* OpenAI API quota limitations, which prevented remote LLM-based chunk enrichment

To keep the assignment functional and complete, a local contextual enrichment strategy was used instead. This allowed the pipeline to remain faithful to the contextual retrieval idea while avoiding API dependency.

The final chatbot uses the **Contextual Retrieval** backend, as required by the assignment, and shows the source chunk used for each answer.


## Example Chatbot Behavior

The chatbot supports free-text questions such as:

* What is lexical semantics?
* What is word2vec?
* What is cosine similarity?
* How can vector models be evaluated?

For each question, the app returns:

* the generated answer
* the source chunk used from Chapter 5



## Conclusion

This assignment successfully implemented both **Naive RAG** and **Contextual Retrieval** for Chapter 5 (*Embeddings*).

The project demonstrated the complete workflow of document preprocessing, chunking, embedding, retrieval, QA construction, evaluation, and web deployment. Although the contextual enrichment approach did not outperform Naive RAG in this local implementation, the experiment provided useful insight into how chunk enrichment quality affects retrieval performance.

Overall, the assignment shows that retrieval-based QA pipelines can be built effectively on textbook chapters, and that evaluation is essential when comparing different retrieval strategies.


