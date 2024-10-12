import boto3
from langchain_aws import BedrockLLM
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

load_dotenv()

# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate
)

# Create a Bedrock Runtime client in the AWS Region you want to use.
client = boto3.client("bedrock-runtime", region_name="ap-south-1")

# Set the model ID, e.g., Titan Text Premier.
model_id = "amazon.titan-text-express-v1"

# Correct prompt initialization using ChatPromptTemplate.from_template
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a nice chatbot having a conversation with a human."
        ),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize the Bedrock LLM with parameters
llm = BedrockLLM(client=client, model_id=model_id, max_tokens=200, temperature=0.7)

# Define a conversation chain with memory
llm_model = LLMChain(
    llm=llm,
    prompt=prompt,  # Correct prompt passed here
    memory=memory
)
# llm_model=llm|prompt|memory

# Main loop for user interaction
while True:
    user_message = input("Please ask your question: ")
    # Ensure that the input is passed correctly
    response = llm_model.invoke({"question": user_message})
    memory.chat_memory.add_user_message(user_message)
    memory.chat_memory.add_ai_message(response["text"])
    print(response["text"])
    print("------------------------------------------------------------------------------------------------------------")
