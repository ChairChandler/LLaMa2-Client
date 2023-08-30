from .llama import LlamaPipeline


from typing import cast

# bug: AttributeError: 'super' object has no attribute '_MemoryLlamaPipelinerun_pipeline'
# solution: translate def __private_method() to def private_method() instead

# bug: problem with inheritance


class MemoryLlamaPipeline:
    def __init__(self, system_prompt: str = '', *args, **kwargs):
        self.super = LlamaPipeline(*args, **kwargs)
        self.chat_history = ''
        self.system_prompt = system_prompt

    def generate_response(self, text_input: str) -> str:
        text = self.parse_input(text_input)
        out = self.super.run_pipeline(text)
        out = cast(list[dict], out)
        out = self.super.postprocess_response(out[0]['generated_text'])

        self.chat_history += f'{out} '
        return out

    def parse_input(self, text_input) -> str:
        if len(self.system_prompt) == 0:
            text_input = self.super.parse_input(text_input)
        else:
            text_input = f'''\
            [INST] <<SYS>>
            {self.system_prompt}
            <</SYS>>
            
            {text_input} [/INST]\
            '''

        self.chat_history += text_input
        return self.chat_history
