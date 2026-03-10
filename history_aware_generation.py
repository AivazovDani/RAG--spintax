from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document


load_dotenv()


# Connect to your document database / Loading the existing Vector DB
persistent_directory = "db/chroma_db"

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)


# Set up Ai model
model = ChatOpenAI(model="gpt-4o")

# Store our conversation as messages
chat_history = []

# Store the conversation number 
convo_number = 1


def learn_from_convo(value1, answer):
    # Create a new langchain document
    new_doc = Document(
        page_content = f"Question: {value1}\nAnswer: {answer}",
        metadata={"source": "conversation"}
    )


    # split the document into chinks
    splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=0)
    chunks = splitter.split_documents([new_doc])

    # add to the vector db which already converts them to vectors in our ingestion pipeline
    db.add_documents(chunks)

def save_conversations(value1, answer):
    global convo_number
    with open("docs/conversation-history.txt", "a") as f:
        f.write(f"This is conversation number {convo_number}\n")
        f.write(f"Question: {value1}\n")
        f.write(f"Answer: {answer}\n")
        convo_number += 1

def ask_question(value1):
    # Step 1: Make the question clear using conversation history
    if chat_history:

        # Ask Ai to make the question standalone
        message = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question")
        ] + chat_history + [
            HumanMessage(content=f"New question: {value1}")
        ]

        result = model.invoke(message)
        search_question = result.content.strip()
    else:
        search_question = value1

    # Step 2: Find relevant documents
    retriever = db.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(search_question)

    for i, doc in enumerate(docs, 1):
        # Show first 2 lines of each document
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f" Doc {i}: {preview}.....")
    
    # Step 3: Create final promt
    combined_input = f"""Based on the following documents and your real reasoning, please answer this question: {value1}

    Documents:
    {"\n".join([f"- {doc.page_content}" for doc in docs])}

    Please provide a clear, helpful answer using the information from these documents and your reasining abilities."
    """
    
    # Step 4: Get the answer
    messages = [
        SystemMessage(content="""You are an SMS copywriter specialized in the Payday lending vertical.
                        You write spintexts using soft transactional language.
                        Never use: loan, guaranteed approval, interest rates, borrow money, payday loan, no credit check.
                        Always use: transfer, deposit, funding, pre-approved, advance, credit, payout, access funds.
                        When generating copies always use spintext format with {option1|option2|option3} syntax.
                        Always add [link1] at the end of every copy.
                        Keep copies under 148 characters excluding the link.
                        Learn from the provided examples and match their structure, tone and urgency level."""),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]
    
    result = model.invoke(messages)
    answer = result.content
    
    # Step 5: Remember this conversation
    chat_history.append(HumanMessage(content=value1))
    chat_history.append(AIMessage(content=answer))
    learn_from_convo(value1, answer)
    save_conversations(value1, answer)

    
    print(f"Answer: {answer}")
    return answer
    start_chat()