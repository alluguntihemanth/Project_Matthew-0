import streamlit as st
import os
from groq import Groq
from langchain.chains import ConversationChain, LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

def main():
    # Get Groq API key
    groq_api_key = os.environ['GROQ_API_KEY']

    # Display the Groq logo
    spacer, col = st.columns([5, 1])  
    with col:  
        st.image('groqcloud_darkmode.png')

    # The title and greeting message of the Streamlit application
    st.title("Chat with Groq!")
    st.write("Hello! I'm your friendly Groq chatbot. Let's start our conversation!")

    # Add customization options to the sidebar
    st.sidebar.title('Customization')
    system_prompt = st.sidebar.text_input("System prompt:")
    model = st.sidebar.selectbox(
        'Choose a model',
        ['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it']
    )
    conversational_memory_length = st.sidebar.slider('Conversational memory length:', 1, 10, value=5)

    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Callback function to clear input after submission
    def clear_input():
        st.session_state.user_input = ""

    # Input for user's question with the callback
    user_input = st.text_input("Ask a question:", key="user_input", on_change=clear_input)

    if st.button("Submit"):
        if user_input:
            # Construct the conversation
            groq_chat = ChatGroq(
                groq_api_key=groq_api_key, 
                model_name=model
            )

            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{human_input}")
            ])

            conversation = LLMChain(
                llm=groq_chat,
                prompt=prompt,
                verbose=True,
                memory=memory,
            )

            response = conversation.predict(human_input=user_input)
            st.session_state.chat_history.append({'human': user_input, 'AI': response})

            # Clear the input box for the next question using the callback
            clear_input()
            st.experimental_rerun()

    # Display chat history
    for message in st.session_state.chat_history:
        st.write(f"**You:** {message['human']}")
        st.write(f"**Chatbot:** {message['AI']}")

if __name__ == "__main__":
    main()


