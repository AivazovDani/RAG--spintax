# RAG--spintax
Web App for Rotating Copies with Ai Memorization. The concept of this app is not complicated, and it’s the preatty simple. The concept is to get an AI API with memory turned on, so we can use it in a simple input output field in our web app.


Definition: a method where we combine LLMs with a retrieval system. This retrieval system can search through vast sources of external info - like documents, databases, etc, whenever the LLM needs additional knowledge to give you better answers.

Context Window - amount of information an LLM can proccess at a given time

What are tokens - a unit of text that the LLM processes. LLMs have a window on how many tokens they can handle.

RAG - > Ingestion Pipeline - > Retrieval Pipeline

Ingestion Pipeline: Source / Documents (Entire knowledge base)> Chunking (breaking down documents into smaller parts) - > To figure out how many chunks we will have, we divide the Source / Documents (Entire knowledge base) by the tokens that we want one chunk to have. Example 10 million// 1 000 = 10 000 chunks - > Then we need to pass the chunks into an embedding model (converts English words, phrases, or paragraphs into a vector representation (math representation)

What are Embeddings - > A vector embedding is a mathematical representation of words, sentances or even images. For example, the word “cat” can have a vector embedding that can look something like this: [34, 21, 7.5]. The numbers are called dimensions. Words with similar semantic meaning tend to have dimensions that are closer to each other.

Popular Embedding Models:

- OpenAi

Vector DB - > a database built to store all the vector embeddings. To store all the vectors, we can use - Pinecone, Weaviate, ChromaDB, and FAISS.

Retrieval Pipeline:

Query (question to the RAG system) - > Vector Embedding (again) that converts the query into X amount of vector embeddings - > Retriever ( basically, from my understanding, this is a step that allows us to match the vector embeddings from our previous step to our vector embeddings in the Ingestion Pipeline. Then it searches by similarity and shortlists the matches, and we can instruct it to give us the top 5 most relevant chunks from the vector DB based on the user query. Then lastly, we send that info to the LLM. We like to tell the LLM this is the user's query, these are all the chunks that might be answers to the query, and figure out the answer.

You must use the same embedding model for both your documents and your user queries.

￼

How the Retriever works: Cosine Similarity:

Cosine similarity measures the angle between vectors, not their magnitude. The cosine similarity values range from 0 to 1, with 0 being the least similar and 1 being the most similar.

Formula - > (A * B) / ( || A || x || B || )

Example: Vector A = [0.6, 0.3, 0.2, 0.7]
Vector B = [0.7, 0.4, 0.1, 0.6]

Then we multiply Vector A's first dimension with Vector B's first dimension, and we do that for every Dimension of the 2 vectors. Then we combine “+” the произведение, and this is our Similarity between the 2 vectors, cause this ( || A || x || B || ) in magnitude is always 1


History-aware generation:

In basic RAG, each query is treated independently. The retriever takes your exact question and searches for chunks. In history-aware RAG, there’s one crucial extra step: query reformulation. Before searching, the system looks at the conversation and rewrites vague or context-dependent questions into clear, standalone questions.

RAG-chunking Strategies:

RAG Ingestion Pipeline:
- Document Loading
- Text Chunking (long text -> smaller pieces)
- Embedding (text chunks -> vectors)
- Storage (vectors -> vector database

Chunks are the most important in our system. The problem with Basic Chunking (CharacterTextSplitter) -> it just cuts text at a fixed character count. Simple, but crude (суров). The problem is that the context can get lost, cause we can split a crucial part into 2. After all, we split per 1000 tokens, for example.

1) CharacterTextSplitter
- splits on diffrent pattern (it follows a split-first, merge-second approach. Split - break text at separators (default: \n\n). Merge - combine pieces until hitting chunk size limit (we can set it ourselves)
- still useful for simple documents

2) RecursiveCharacterTextSplitter
- splits at natural bounderies (<p>, sentance, words). Basically breaks up long documents into meaningful pieces by finding where topics naturally change. Instead of using a random words counter, it uses AI embeddings to understand the semantic meaning of sentences

3) Document SpecificSplitting
- Respect the document structure

4) SemanticSplitting
- uses embeddings to detect topic shifts
- keeps related concepts together
- splits when meaning changes, not just by size

5) Agentic Splitting
- The LLM itself analyses content and uses an optimal split

Document Loading:

Atomic Element - > minimal, self-contained, and non-divisible piece of information extracted from a larger document or text chunk

LangChain Document - > converts data from various formats (csv, pdf, HTML) into standardized document objects for LLM applications. Really important in our.

Each document object has 2 fields:

Document( page_content="The raw text content of the file...", metadata={"source": "docs/your_file.txt"} 


load_dotenv() is a function from the python-dotenv library used to read key-value pairs from a .env file and set them as environment variables in os.environ

glob is a pattern-matching syntax for file paths:

“*.txt” - > all the top-level txt files in the folders 

Splitting Documents:

CharacterTextSplitter tries to split on these characters in order of preference:
"\n\n" → paragraph breaks (preferred)
"\n" → line breaks
" " → spaces
→ character by character (last resort)

chunk overlap - the practice of repeating a small, specified portion of text (characters or tokens) from the end of one chunk at the beginning of the next. It acts as a buffer to maintain semantic context across document boundaries, ensuring that critical information split by the chunker is not lost.


Chroma DB:

Chroma DB has 2 states - in-memory only (lost when the script ends) and persisted to disk.
With `persist_directory`, Chroma writes its data to disk. This means you run the ingestion pipeline once, and then your retrieval app just loads the existing DB — you don't re-embed everything every time.

HNSW stands for Hierarchical Navigable Small World - it's the algorithm Chroma uses internally to search for nearest neighbors efficiently without comparing against every single vector. The `cosine` space tells it to measure similarity using cosine similarity, which measures the angle between two vectors rather than the distance between them

relevant_docs = retriever.invoke(query)

What does this line do?

This single line does three things invisibly:

1. Takes your query string: "How machine learning helps humanity."
         │
         ▼
2. Sends it to OpenAI Embeddings API → converts to vector [0.21, -0.54, ...]
         │
         ▼
3. Searches Chroma for the 5 closest vectors → returns List[Document]

How to include memory in our model:

The set-up with Loading the existing Vector DB is standard. Then we set up the AI model and create a history, a list - chat_history to store our conversation. Then we create a function ask_question. In this function, we have 5 steps. Step 1 - make the conversation clear using chat history. Step 2: 
Find the top 5 answers and get them. Step 3: Create the final prompt. Step 4: Get the answer. And Step 5: Remember the conversation by appending to the chat history

First, we check if there is something in the chat_history, cause when we ask the first question, there is nothing to get from out chat_historty. Then, on the second question, we use message = [SystemMessage + chat_history + Human Message]. This is done because when we ask a question like Tell me more about that the LLM doesn’t have “that” in the DB, so it needs a system config (SystemMessage) + the previous chat history + the current question from us (HumanMessage). This way, later below, it rewrites the question for itself to understand it and respond accordingly. After that message = [], we make an AIMessage object and take the content inside it (we take the text from the AI). Then we need to find the top 5 best-matching chunks and then use this line, which does 3 things at once: relevant_docs = retriever.invoke(query). Then we create the final prompt using the user_question, and then we list out all the chunks with the highest similarity to our question. Then we start a new message = [] where we don’t want t stand alone answer, so we can use it to find the relevant chunks (as above) and create the actual prompt. Here, we want the answer to our question, and the HumanMessage is our actual (combined_input) prompt. Then we again create an AIMEssage object and get the text from the AI output. Then we append the HumanMessage and the AIMessage to our chat_history and return the answer.


Now we want to input the actual data into our AI model. We’ll have to update the content inside those docs using a cron job.

How to structure the docs: You have the vertical type (example: Payday) with a brief explanation about what Payday is, then you have a list of winner copies and a breakdown of each (like a pattern), so the AI knows what pattern to follow when creating the copies for each vertical.

Can I save every convo we have, so it learn more?

A langchain document is a class with two fields: page_content (the page content) and metadata (the source, author, date, etc.). In the vector database, the document get splits into a unique ID, vectors  (the embedded version of the page_content), and the document itself with the page_content and metadata.


db.add_documents(chunks) - >

Step 1 - EMBED

- Takes each chunk's page_content
- sends it to OpenAI Embeddings API
- gets back vectors [0.21, -0.54, 0.87...]
        │
        ▼
Step 2 - COMBINE

- Pair each vector with its
- original text + metadata
        │
        ▼
Step 3 - STORE

- Writes everything into the
- existing Chroma DB on disk



Adding to existing DB vs creating an ingestion pipeline:

INGESTION PIPELINE             ADDING TO EXISTING DB
────────────────    ────────────────────
runs ONCE at the start                runs DYNAMICALLY anytime

1. load_documents()                   1. Create Document object
   - reads .txt files                            - manually
        │                                                    │
        ▼                                                   ▼
2. split_documents()                    2. split_documents()
   - chunks the text                          - chunks it
        │                                                    │
        ▼                                                   ▼
3. Chroma.from_documents()      3. db.add_documents()
   - CREATES new DB                   - ADDS to existing DB
   - embeds everything                     - embeds new chunks only
   - saves to disk                               - saves to disk 
