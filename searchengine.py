#importing important libraries
import streamlit as st
import os
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv

load_dotenv()

##arxive, wikipedia and searchtool
arxiv_api_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
arxiv = ArxivQueryRun(api_wrapper= arxiv_api_wrapper)

wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wiki = WikipediaQueryRun(api_wrapper = wiki_api_wrapper)

search = DuckDuckGoSearchRun(name="search")


#title
st.title("Langchain-chat with search")


#side bar for settings
st.sidebar.title("settings")
api_key = st.sidebar.text_input("Enter your Groq Api key:", type="password")

if "messages" not in st.session_state:
    st.session_state ["messages"]=[
        {"role":"assistent", "content":"Hi!, i am a chatbot who can search the web. How can i help you?"}
    ]


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


if prompt:=st.chat_input(placeholder="what is machine learning?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)
    
    llm=ChatGroq(groq_api_key = api_key, model_name="Llama3-8b-8192")

    #combine tools
    tools = [arxiv, wiki, search]

    #convet tools into agent so that we can invoke it
    search_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)

    with st.chat_message("assistent"):
        st_cd = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        response = search_agent.run(st.session_state.messages, callbacks=[st_cd])

        st.session_state.messages.append({"role":"assistent", "content":response})

        st.write(response)









