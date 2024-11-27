import chainlit as cl
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
import warnings
import asyncio

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", message=".*torch.utils._pytree.*")

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV")
index_name = "new"

# Load embeddings
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone
pinecone_instance = PineconeClient(api_key=PINECONE_API_KEY)
if index_name not in [index.name for index in pinecone_instance.list_indexes()]:
    pinecone_instance.create_index(
        name=index_name,
        dimension=768,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region=PINECONE_API_ENV)
    )

# Access the index
docsearch = Pinecone.from_existing_index(index_name, embeddings)

# Define prompt template
PROMPT = PromptTemplate(template="Answer the following question: {context}", input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

# Define Chat Profiles
@cl.set_chat_profiles
async def chat_profiles():
    return [
        cl.ChatProfile(
            name="LLAMA",
            markdown_description="A versatile assistant capable of handling general inquiries.",
        ),
        cl.ChatProfile(
            name="Mistral",
            markdown_description="An assistant specialized in providing technical support.",
        )
    ]

# Initialize the LLM
def initialize_llm(profile_name):
    model_path = "model/llama-2-13b-chat.ggmlv3.q4_0.bin"  # Default model path
    if profile_name == "Technical Support":
        model_path = "model/tech_support_model.bin"
    elif profile_name == "Sales Advisor":
        model_path = "model/sales_advisor_model.bin"
    # Initialize the LLM with the selected model
    return CTransformers(
        model=model_path,
        model_type="llama",
        config={
            'max_new_tokens': 512,
            'temperature': 0.5
        }
    )

# Truncate context to fit within the LLM's token limit
def truncate_context(context, max_tokens=512):
    return context[:max_tokens]

# Define synchronous long-running task
def long_running_task(input_data, llm):
    """Long-running synchronous task for LLM inference."""
    print("Generating response...")
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )
    result = qa.invoke(input_data)
    print("Response generated:", result["result"])
    return result["result"]

# Convert the synchronous function to an asynchronous one
async_long_running_task = cl.make_async(long_running_task)

@cl.on_chat_start
async def start_chat():
    """Send a welcome message when the chat starts."""
    await cl.Message(content="Welcome! Please select a chat profile to begin.").send()

@cl.on_chat_resume
async def on_chat_resume():
    """Handle chat session resumption."""
    try:
        # Retrieve the last selected chat profile from the user session
        chat_profile = cl.user_session.get("chat_profile")

        if chat_profile:
            # Inform the user about the resumed chat profile
            await cl.Message(content=f"Welcome back! Resuming your session with the '{chat_profile}' profile.").send()
        else:
            # Prompt the user to select a profile if not set
            await cl.Message(content="Welcome back! Please select a chat profile to continue.").send()
    except Exception as e:
        print(f"Error in on_chat_resume: {e}")
        await cl.Message(content="An error occurred while resuming your session. Please try again.").send()

@cl.on_message
async def handle_message(message):
    """Handle user messages and process the query."""
    try:
        print("Message received from UI:", message.content)
        query = message.content

        # Retrieve the selected chat profile
        chat_profile = cl.user_session.get("chat_profile")
        if not chat_profile:
            await cl.Message(content="Please select a chat profile to proceed.").send()
            return

        # Initialize the LLM based on the selected chat profile
        llm = initialize_llm(chat_profile)

        # Send an initial "processing" message
        processing_message = await cl.Message(content=f"Processing your request with the '{chat_profile}' profile, please wait...").send()

        # Retrieve documents
        print("Retrieving documents...")
        retriever = docsearch.as_retriever(search_kwargs={'k': 2})
        docs = retriever.invoke(query)

        # Combine and truncate context
        print("Processing retrieved documents...")
        context = " ".join([doc.page_content for doc in docs])
        truncated_context = truncate_context(context, max_tokens=512)

        # Prepare input data for the LLM
        input_data = {"query": query, "context": truncated_context}

        # Send periodic progress updates
        for i in range(3):  # Send 3 periodic updates
            await asyncio.sleep(5)  # Simulate processing delay
            await cl.Message(content=f"Processing... {i * 33}% done").send()

        # Call the asynchronous long-running task
        result = await async_long_running_task(input_data, llm)

        # Update the user with the result
        await cl.Message(content=result).send()
        print("Response sent successfully!")
    except Exception as e:
        print(f"Error occurred: {e}")
        await cl.Message(content="An error occurred while processing your request. Please try again.").send()
