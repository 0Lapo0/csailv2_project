import streamlit as st
import chromadb
from groq import Groq
from pypdf import PdfReader
API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=API_KEY)
models = ("openai/gpt-oss-120b", "llama-3.1-8b-instant", "llama-3.3-70b-versatile", "openai/gpt-oss-20b" )
MODEL = st.sidebar.selectbox("choose model", models)
# Initialize
if "messages" not in st.session_state:
    st.session_state.messages = []
if "client" not in st.session_state:
    st.session_state.client = chromadb.Client()
if "collection" not in st.session_state:
    try:
        st.session_state.client.delete_collection("documents")
    except:
        pass
    c = st.session_state.client.create_collection("documents")
    st.session_state.collection = c.name
if "collections" not in st.session_state:
    st.session_state.collections = []
if "overlap" not in st.session_state:
    st.session_state.overlap = 0
if "chunk_size" not in st.session_state:
    st.session_state.chunk_size = 0
if "step" not in st.session_state:
    st.session_state.step = 0
if "used_files" not in st.session_state:
    st.session_state.used_files = []
if "processed" not in st.session_state:
    st.session_state.processed = []
with (st.sidebar):
    old_over, old_size = st.session_state.overlap, st.session_state.chunk_size
    st.session_state.chunk_size = st.slider("chunk size", min_value=0, max_value=1000, value=400)
    st.session_state.overlap = st.slider("overlap", min_value=0, max_value=1000, value=70)
    st.session_state.step = st.session_state.chunk_size - st.session_state.overlap
    if old_over != st.session_state.overlap or old_size != st.session_state.chunk_size:
        for collection in st.session_state.client.list_collections():
            st.session_state.client.delete_collection(collection.name)
        st.session_state.client.create_collection("documents")
        st.session_state.collections = []
        st.session_state.processed = []
        st.session_state.used_files = []
        st.toast("collection reset")
        st.rerun()
        #st.write(st.session_state.collection)

    types = [
        "txt",
        "md",
        "rst",
        "py",
        "js",
        "html",
        "css",
        "json",
        "xml",
        "csv",
        "yaml",
        "ini",
        "conf",
        "env",
        "svg",
        "pdf"
    ]
    files = st.file_uploader("Upload file", type=types, accept_multiple_files=True)
    c1, c2 = st.columns(2)
    with c1:
        if files and st.button("process file"):
            for file in files:
                if file.name not in st.session_state.processed:
                    st.toast("file processed")
                    t_types = {
                        "text/plain",
                        "text/markdown",
                        "text/x-rst",
                        "text/x-python",
                        "text/javascript",
                        "text/html",
                        "text/css",
                        "application/json",
                        "application/xml",
                        "text/csv",
                        "application/x-yaml",
                        "text/x-ini",
                        "image/svg+xml"
                    }
                    if file.type in t_types:
                        text = file.read().decode("utf-8")
                    elif file.type == "application/pdf":
                        reader = PdfReader(file)
                        text = ""
                        for page in reader.pages:
                            text += page.extract_text() + "\n"
                    #highly customisable
                    chunks = []
                    for i in range(0, len(text), st.session_state.step):
                        chunks.append(text[i: i + st.session_state.chunk_size])
                    for i in range(len(chunks)):
                        chunks[i] = f"from file: {file.name}: {chunks[i]}"
                    tags = [file.name + str(i) for i in range(len(chunks))] #add file labeling
                    collection = st.session_state.client.create_collection(file.name)
                    collection.add(documents=chunks, ids=tags)
                    st.session_state.collections.append(collection.name)
                    st.session_state.processed.append(file.name)
                    st.toast("chunks added")

        if len(st.session_state.collections) != 0:
            for collection in st.session_state.collections:
                coll = st.session_state.client.get_collection(collection).get()
                if st.checkbox(f"use {collection}", value=True):
                    if collection not in st.session_state.used_files:
                        st.session_state.used_files.append(collection)
                        st.session_state.client.get_collection(st.session_state.collection).add(documents=coll["documents"], ids=coll["ids"])
                else:
                    if collection in st.session_state.used_files:
                        st.session_state.used_files.remove(collection)
                        st.session_state.client.get_collection(st.session_state.collection).delete(ids=coll["ids"])
    with c2:
        if st.button("reset context memory"):
            for collection in st.session_state.client.list_collections():
                st.session_state.client.delete_collection(collection.name)
            #st.session_state.client.delete_collection("documents")
            st.session_state.client.create_collection("documents")
            st.session_state.collections = []
            st.session_state.processed = []
            st.session_state.used_files = []
            st.toast("collection reset")
            #st.write(st.session_state.collection)
            st.rerun()
with st.bottom:
    col1, col2 = st.columns(2)
    fixed_question = ""
    with col1:
        question = st.text_input("wanna chat?")
        fix = st.checkbox("fix the question?")
        if fix:
            translate = [{"role": "system",
                          "content": "if nothing is wrong do not change anything. do not add anything, just return the fixed question by itself. use the closest word when translating."},
                         {"role": "user",
                          "content": f"translate the following question to english if it is not already, and fix any major spelling mistakes: {question}"}]
            fixed_question = client.chat.completions.create(model=MODEL, messages=translate).choices[0].message.content
            #st.write(fixed_question)
        history = st.checkbox("use chat history? (might take longer to asnwer)")
    with col2:
        st.space()
        if st.button("send") and question:
            if fix:
                que = fixed_question
            else:
                que = question
            #st.write(que)
            collection = st.session_state.client.get_collection("documents")
            result = collection.query(query_texts=que, n_results=5)
            # st.session_state.context = result["documents"][0][::-1] #add thresholding
            st.session_state.context = []
            for i in range(len(result["documents"][0])):
                if result["distances"][0][i] < 1.5:
                    st.session_state.context.append(result["documents"][0][i])
            st.session_state.distances = result["distances"][0]
            st.session_state.question = que
            #for ans in st.session_state.context:
            #    st.write(ans)
            #st.write(st.session_state.distances)
            context = "\n".join(st.session_state.context)
            question = st.session_state.question
            messages = [{"role": "system",
                         "content": "Answer the user's question using only the provided document context.If the context contains enough information to answer, give the answer. use previous answers as a way to understand the question, but do not use any previous context as a source to answer a question that does not directly ask about a previous question.."},
                        {"role": "user", "content": f"DOCUMENT CONTEXT:\n{context}\n\nQUESTION:\n{question}"}]
            st.session_state.messages.append(messages[0])
            st.session_state.messages.append(messages[1])
            if history:
                response = client.chat.completions.create(model=MODEL, messages=st.session_state.messages)
            else:
                response = client.chat.completions.create(model=MODEL, messages=messages)
            st.session_state.messages.append({"role":"assistant", "content":response.choices[0].message.content})
        with st.expander("view fixed question"):
            st.write(fixed_question)
        if st.button("clear history"):
            st.session_state.messages.clear()
for message in st.session_state.messages:
    if message["role"] == "assistant":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    elif message["role"] == "user":
        with st.chat_message(message["role"]):
            st.markdown(message["content"].split("QUESTION:\n", 1)[1])