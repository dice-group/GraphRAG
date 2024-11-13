# GraphRAG Setup and Usage Guide
## 1. Install GraphRAG
To begin using GraphRAG, you first need to install it using pip. If you encounter any installation errors, follow the instructions below.

# Install GraphRAG
```pip install graphrag```


## 2. Initialize GraphRAG
After installation, you need to initialize GraphRAG. This will set up the necessary directories and files.

```python -m graphrag.index --init --root .```

This command will create the .env file where you can configure your environment variables and set up your project.

## 3. Configure OPENAI_API_KEY
GraphRAG uses the OpenAI API for generating responses, so you will need to configure your OPENAI_API_KEY in the .env file. After initializing GraphRAG, an ```.env``` file will be created in the root directory. Edit the file to include your API key:

```OPENAI_API_KEY=your_openai_api_key_here```

## 4. Run GraphRAG to Create Graphs on the Data
Once the setup is complete, you can start the process of creating entity and relationship graphs from your text data. Run the following command to index your data and generate the necessary Parquet files:

```python -m graphrag.index --root .```

This command will process the input data and create the graph files required for the Retrieval-Augmented Generation (RAG) system.

5. Evaluate the RAG Model
Once the graphs are created, you can evaluate the RAG model by querying it. There are two types of queries you can use: local and global.

Global search: Searches across all available data.

```python -m graphrag.query --root . --method global --query "What are the main themes?"```

Local search: Searches specific to a certain entity or dataset.

```python -m graphrag.query --root . --method local --query "Tell me about specific entity ```
