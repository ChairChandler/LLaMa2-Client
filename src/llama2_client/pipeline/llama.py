from typing import cast
from optimum.bettertransformer import BetterTransformer
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
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding=True
        )
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # TODO Optimization -> https://github.com/huggingface/accelerate/issues/483
        self.model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16 if low_memory else torch.float32,
            offload_state_dict=low_memory,
            max_memory={'cpu': f'{maximum_memory_size_gb}GB'}
        )
        self.model.resize_token_embeddings(len(self.tokenizer))  # type: ignore
        self.model = BetterTransformer.transform(self.model)  # type: ignore
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
        text = self.parse_input(text_input)
        out = self.run_pipeline(text)
        out = cast(list[dict], out)
        out = self.postprocess_response(out[0]['generated_text'])
        return out

    def parse_input(self, text_input) -> str:
        return fr'[INST]{text_input}[/INST]'

    def run_pipeline(self, text_input):
        return self.pipe(
            text_input,
            do_sample=True,
            top_k=self.top_k_beams,
            top_p=self.top_p,
            temperature=self.temperature,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            min_new_tokens=self.min_output_length,
            max_new_tokens=self.max_output_length,
            use_cache=True
        )

    @staticmethod
    def postprocess_response(txt: str) -> str:
        search_txt = '[/INST]'
        pos = txt.rfind(search_txt)
        offset = len(search_txt) + 1  # space after text
        return txt[pos+offset:]
