import psutil
from typing import cast
import streamlit as st

from .pipeline import MemoryLlamaPipeline

HUGGINGFACE_MODELS = [
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf"
]

st.title("💬 LLaMa2 Chatbot")

# left panel

with st.sidebar:
    st.header('Model settings')

    selected_model = st.selectbox(
        'LLaMa2 model',
        HUGGINGFACE_MODELS,
        index=0
    )

    low_memory = st.checkbox(
        'Low memory',
        value=True
    )

    max_sys_size = psutil.virtual_memory().total // (1024**3)
    maximum_memory_size = st.number_input(
        'Maximum memory (GB)',
        min_value=0,
        max_value=max_sys_size,
        value=round(0.5*max_sys_size)
    )

    output_length = st.slider(
        'Output length',
        min_value=0,
        max_value=1024,
        value=(1, 64)
    )
    min_length, max_length = output_length

    top_k_beams = st.number_input(
        'Top-k beams',
        min_value=1,
        value=3
    )

    top_p = st.number_input(
        'Top-p',
        min_value=0.,
        max_value=1.,
        step=0.01,
        value=0.9
    )

    temperature = st.number_input(
        'Temperature',
        min_value=0.,
        max_value=100.,
        step=0.1,
        value=0.7
    )

    # system prompt
    sys_prompt = st.text_area('System prompt')

    model_reloading = st.button('Reload model with settings')

# model (re)loading & session state init


def is_first_time_loading(): return "messages" not in st.session_state


if is_first_time_loading() or model_reloading:
    st.session_state["sys_prompt"] = sys_prompt
    st.session_state["messages"] = []
    st.session_state["history"] = []
    with st.spinner('Model loading'):

        if "model" in st.session_state:
            del st.session_state['model']
        st.session_state['model'] = MemoryLlamaPipeline(
            model_name=cast(str, selected_model),
            maximum_memory_size_gb=cast(int, maximum_memory_size),
            min_output_length=min_length,
            max_output_length=max_length,
            top_k_beams=cast(int, top_k_beams),
            top_p=cast(int, top_p),
            temperature=cast(int, temperature),
            low_memory=low_memory,
            system_prompt=st.session_state.sys_prompt
        )

# chat

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    response = st.session_state.model.generate_response(prompt)
    st.session_state.messages.append(
        {"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)

    st.session_state.history.append()
