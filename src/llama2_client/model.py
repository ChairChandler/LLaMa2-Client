from typing import cast
from typing_extensions import override
# from optimum.bettertransformer import BetterTransformer
from transformers import pipeline, LlamaForCausalLM, AutoTokenizer, PreTrainedModel
import torch

class LlamaPipeline:
    def __init__(
            self, 
            model_name: str, 
            maximum_memory_size_gb: int,
            min_output_length: int, 
            max_output_length: int,
            top_k_beams: int,
            top_p: int,
            temperature: int,
            low_memory=True
        ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name, 
            low_cpu_mem_usage=True, 
            torch_dtype=torch.bfloat16 if low_memory else torch.float32,
            max_memory={'cpu': f'{maximum_memory_size_gb}GB'}
        )
        self.model.resize_token_embeddings(len(self.tokenizer)) # type: ignore
        # self.model = BetterTransformer.transform(self.model)
        self.pipe = pipeline(
            "text-generation", 
            model=cast(PreTrainedModel, self.model), 
            tokenizer=self.tokenizer
        )
        self.min_output_length = min_output_length
        self.max_output_length = max_output_length
        self.top_k_beams = top_k_beams
        self.temperature = temperature
        self.top_p = top_p
    
    def generate_response(self, text_input: str) -> str:
        text = self.__parse_input(text_input)
        out = self.__run_pipeline(text)
        out = cast(list[dict], out)
        out = self.__postprocess_response(out[0]['generated_text'])
        return out

    def __parse_input(self, text_input) -> str:
        return fr'[INST]{text_input}[/INST]'

    def __run_pipeline(self, text_input):
        out = self.pipe(
            text_input,
            do_sample=True,
            top_k=self.top_k_beams,
            top_p=self.top_p,
            temperature=self.temperature,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            min_length=self.min_output_length,
            max_length=self.max_output_length
        )
        
        return out
    
    @staticmethod
    def __postprocess_response(txt: str) -> str:
        search_txt = '[/INST]'
        pos = txt.find(search_txt)
        offset = len(search_txt) + 1 # space after text
        out = txt[pos+offset:]
        return out
    
class MemoryLlamaPipeline(LlamaPipeline):
    def __init__(self, system_prompt: str = '', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chat_history = ''
        self.system_prompt = system_prompt
        
    def generate_response(self, text_input: str) -> str:
        text = self.__parse_input(text_input)
        out = self.__run_pipeline(text)
        out = cast(list[dict], out)
        out = self.__postprocess_response(out[0]['generated_text'])
        
        self.chat_history += out + ' ' # space after model text
        return out
        
    @override
    def __parse_input(self, text_input) -> str:
        if self.chat_history:
            text_input = super().__parse_input(text_input)
        else:
            text_input = f'[INST] <<SYS>>\n\
            {self.system_prompt}\n\
            <</SYS>>\n\
            \n\
            {text_input} [/INST]'
            
        self.chat_history += text_input
        return self.chat_history
            
    