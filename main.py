import streamlit as st
from streamlit_chat import message
import os
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts.prompt import PromptTemplate

os.environ['API_KEY'] = st.secrets['API_KEY']

template = """You are a Scienctist that spechializes in earthquakes and natural disasters \
and you are also a helpful friend that talks with people and help them \
you also are good at general chat and talking with people and answering questions \
when someone asks you who are you, you say that you are a friend that talks with people \
you don't talk instead of the person you just get person response and answer it for him \
When you don't know the answer to a question you admit\
that you don't know.

and you answer the person's question like this example :
example :

person:what is earthquake
AI:The tectonic plates are always slowly moving, but they get stuck at their edges due to friction. When the stress on the edge overcomes the friction, there is an earthquake that releases energy in waves that travel through the earth's crust and cause the shaking that we feel.

Current conversation:
{history}
Human: {input}
AI Assistant:"""

def main(api_key,template):
    
    llm = ChatGoogleGenerativeAI(google_api_key=api_key,model="gemini-pro",temperature=0.7)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]

    st.header("Your AI Home Assistant 🤖")

    prompt_template = PromptTemplate(input_variables=["history", "input"], template=template)
    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=500)

    conversation = ConversationChain(
        prompt=prompt_template,
        llm=llm, 
        memory = memory,
        verbose=True
    )

    with st.sidebar:
        input = st.text_input("Your message: ", key="input")

        if input:
            st.session_state.messages.append(HumanMessage(content=input))
            with st.spinner("Thinking..."):
                response = conversation.predict(input=st.session_state.messages)

            if isinstance(response, str):
                print(type(response.strip()))
                st.session_state.messages.append(
                    AIMessage(content=response))
            else:
                print("hi")
                st.session_state.messages.append(
                    AIMessage(content=response.content))
            
    messages = st.session_state.get('messages', [])
    for i, msg in enumerate(messages[1:]):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=str(i) + '_user')
        else:
            message(msg.content, is_user=False, key=str(i) + '_ai')

if __name__ == '__main__':
    main(api_key=os.environ['API_KEY'],template=template)
