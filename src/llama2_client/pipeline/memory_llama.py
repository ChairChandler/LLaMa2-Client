from .llama import LlamaPipeline


from typing import cast

# inheritance not working properly in streamlit session state
# bug: AttributeError: 'super' object has no attribute '_MemoryLlamaPipeline__run_pipeline'


class MemoryLlamaPipeline:
    def __init__(self, system_prompt: str = '', *args, **kwargs):
        self.super_pipeline = LlamaPipeline(*args, **kwargs)
        self.chat_history = ''
        self.system_prompt = system_prompt

    def generate_response(self, text_input: str) -> str:
        text = self.__parse_input(text_input)
        out = self.super_pipeline.__run_pipeline(text)
        out = cast(list[dict], out)
        out = self.super_pipeline.__postprocess_response(
            out[0]['generated_text'])

        self.chat_history += f'{out} '
        return out

    def __parse_input(self, text_input) -> str:
        if self.chat_history:
            text_input = self.super_pipeline.__parse_input(text_input)
        else:
            text_input = f'[INST] <<SYS>>\n\
            {self.system_prompt}\n\
            <</SYS>>\n\
            \n\
            {text_input} [/INST]'

        self.chat_history += text_input
        return self.chat_history
