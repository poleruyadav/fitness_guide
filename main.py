from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

from langchain_groq import ChatGroq

# RAG imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# ---------- LOAD API KEY ----------
def load_api_key():
    with open("groqapi.txt", "r") as f:
        return f.read().strip()


# ---------- CREATE VECTOR STORE ----------
def create_vectorstore():
    loader = PyPDFLoader("fitness_guide.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory="db"
    )

    return vectorstore


# ---------- CREATE CHAIN ----------
def create_chain(vectorstore):
    api_key = load_api_key()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are a certified Fitness Coach.

Use the given context to answer the user.

Context:
{context}

Follow:
- Understand goal
- Give simple workout/diet
- Motivate user
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    model = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=api_key,
        temperature=0.7
    )

    parser = StrOutputParser()

    # ✅ UPDATED RAG CHAIN
    def rag_chain(inputs):
        question = inputs["question"]

        # ✅ FIX HERE (NEW METHOD)
        docs = retriever.invoke(question)

        context = "\n".join([doc.page_content for doc in docs])

        response = (prompt | model | parser).invoke({
            "question": question,
            "chat_history": inputs["chat_history"],
            "context": context
        })

        return response

    return rag_chain


# ---------- MEMORY ----------
def get_memory():
    return InMemoryChatMessageHistory()


# ---------- ASK ----------
def ask_question(chain, memory, question):
    response = chain({
        "question": question,
        "chat_history": memory.messages
    })

    memory.add_message(HumanMessage(content=question))
    memory.add_message(AIMessage(content=response))

    return response