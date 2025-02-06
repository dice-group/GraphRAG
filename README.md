# Langchain-RAG Benchmark Generator

A tool for generating benchmark datasets to evaluate RAG (Retrieval-Augmented Generation) systems. This project generates question-answer pairs from input documents, which can be used to assess the performance of RAG implementations.

## Prerequisites

- Python 3.10 or higher
- Conda package manager
- Access to TENTRIS API endpoints

### How Benchmark Dataset is Generated:
The **Giskard Python library** provides **[RAGET (RAG Evaluation Toolkit)](https://docs.giskard.ai/en/latest/open_source/testset_generation/testset_generation/index.html)**, which automatically generates a benchmark dataset. RAGET works by:

- Generating a list of questions, reference answers, and reference contexts directly from the knowledge base of your RAG system.
- Producing test datasets that can evaluate the retrieval, generation, and overall quality of your RAG system.

This includes simple questions, as well as more complex variations (e.g., situational, double, or conversational questions) designed to target specific components of the RAG pipeline.

## Quick Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd langchain-RAG/benchmark_gen
```

2. Create and activate the Conda environment:
```bash
conda env create -f environment.yml
conda activate benchmark_rag
```

3. Create a .env file based on .env.example:
```bash
TENTRIS_API_KEY=your_api_key_here
```
## Usage

### Generate benchmark dataset from a text file:
```bash
python benchmark_dataset.py --input ../data/speech.txt --llm http://tentris-ml.cs.upb.de:8501/v1 --embed http://tentris-ml.cs.upb.de:8502/v1
```

### Using CSV with specific column:
```bash
python benchmark_dataset.py --input ../data/dice_verbalized.csv --csv_source_column content --llm http://tentris-ml.cs.upb.de:8501/v1 --embed http://tentris-ml.cs.upb.de:8502/v1
```

## Command Line Arguments

- `--input`: Path to input document (default: "data/speech.txt")
- `--output`: Path to output JSON file (default: "benchmark_dataset.json")
- `--num_questions`: Number of questions to generate (default: 10)
- `--csv_source_column`: Source column for CSV files
- `--llm`: Base URL for the LLM API
- `--embed`: Base URL for the embedding API


## Output Format
The script generates a JSON file containing:
- Generated questions
- Reference answers
- Context information
- Metadata (question type, seed document, topic)


```
{
  "id": "acc80340-2a9e-469f-96d0-e2ba25d45454",
  "question": "What are Caglar Demir's research interests?",
  "reference_answer": "Caglar Demir's research interests focus on scalable algorithms for learning and reasoning on knowledge graphs.",
  "reference_context": "Caglar Demir is a Senior Researcher at DICE Research. His research interests include scalable algorithms for learning and reasoning on knowledge graphs. He is involved in projects such as LEMUR and SAIL.",
  "conversation_history": [],
  "metadata": {
    "question_type": "simple",
    "seed_document_id": 3,
    "topic": "Others"
  }
}
```
