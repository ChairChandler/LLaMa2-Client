from typing import cast
from optimum.bettertransformer import BetterTransformer
import streamlit as st
from transformers import pipeline, LlamaForCausalLM, AutoTokenizer, PreTrainedModel
import torch

# MODEL

class LlamaPipeline:
    HUGGINGFACE_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
    
    def __init__(self, low_memory=True):
        self.tokenizer = AutoTokenizer.from_pretrained(self.HUGGINGFACE_MODEL_NAME)
        self.model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.HUGGINGFACE_MODEL_NAME, 
            low_cpu_mem_usage=True, 
            torch_dtype=torch.bfloat16 if low_memory else torch.float32,
            max_memory={'cpu': '25GB'}
        )
        # self.model = BetterTransformer.transform(self.model)
        self.pipe = pipeline(
            "text-generation", 
            model=cast(PreTrainedModel, self.model), 
            tokenizer=self.tokenizer
        )
    
    def generate_response(self, text_input: str):
        out = self.pipe(
            fr'{text_input}\n\n',
            do_sample=False,
            top_k=2,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=200
        )
        print(out)
        out = cast(list[dict], out)
        out = self.__postprocess_response(out[0]['generated_text'])
        return out
    
    @staticmethod
    def __postprocess_response(txt: str) -> str:
        search_txt = 'Answer:'
        pos = txt.find(search_txt)
        offset = len(search_txt) + 1 # space after text
        out = txt[pos+offset:]
        return out

# UI

st.title("💬 LLaMa Chatbot")
def remove():
    del st.session_state["messages"]
    del st.session_state['model']
    
low_memory = st.checkbox(
    'Low memory', 
    True, 
    on_change=remove
) 
if "messages" not in st.session_state:
    st.session_state["messages"] = []
    with st.spinner('Model loading'):
        st.session_state['model'] = LlamaPipeline(low_memory)

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    response = st.session_state.model.generate_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)