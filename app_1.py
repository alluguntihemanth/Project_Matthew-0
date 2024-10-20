import streamlit as st
from groq import Groq
import random

from langchain.chains import ConversationChain, LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

def main():
    """
    This function is the main entry point of the application. It sets up the Groq client, the Streamlit interface, and handles the chat interaction.
    """
    
    # Get Groq API key from Streamlit secrets
    groq_api_key = st.secrets["groq"]["api_key"]

    # Display the Groq logo
    spacer, col = st.columns([5, 1])  
    with col:  
        st.image('chatbot-saying-hi.png')

    # The title and greeting message of the Streamlit application
    st.title("Chat with Matthew!")
    st.write("Hello! I'm your friendly conversational-agent. I can help answer your questions, provide information, or just chat. I'm also super fast! Let's start our conversation!")

    # Add customization options to the sidebar
    st.sidebar.title('Customization')
    system_prompt = st.sidebar.text_input("System prompt:")
    model = st.sidebar.selectbox(
        'Choose a model',
        ['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it', 'gemma2-9b-it', 'llama-3.1-70b-versatile']
    )
    conversational_memory_length = st.sidebar.slider('Conversational memory length:', 1, 10, value=5)

    memory = ConversationBufferWindowMemory(
        k=conversational_memory_length, memory_key="chat_history", return_messages=True
    )

    user_question = st.text_input("Ask a question:")

    # Session state variable
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    else:
        for message in st.session_state.chat_history:
            memory.save_context(
                {'input': message['human']},
                {'output': message['AI']}
            )

    # Initialize Groq Langchain chat object and conversation
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name=model
    )

    # If the user has asked a question
    if user_question:
        # Construct a chat prompt template using various components
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),  # Persistent system prompt

                MessagesPlaceholder(variable_name="chat_history"),  # Placeholder for chat history

                HumanMessagePromptTemplate.from_template("{human_input}")  # User's input
            ]
        )

        # Create a conversation chain using LangChain LLM
        conversation = LLMChain(
            llm=groq_chat,
            prompt=prompt,
            verbose=True,
            memory=memory,
        )

        # Generate chatbot's response by sending the prompt to the Groq API
        response = conversation.predict(human_input=user_question)
        message = {'human': user_question, 'AI': response}
        st.session_state.chat_history.append(message)
        st.write("Chatbot:", response)

if __name__ == "__main__":
    main()

