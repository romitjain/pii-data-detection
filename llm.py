"""
Simple wrapper class around GPT API
"""
import sys
import json
import openai
import backoff

from typing import List
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="{time} {level} {message}", level='INFO')

class LLM():
    def __init__(self, model_id: str, messages: List=[], uri: str=None, keep_history: bool=True) -> None:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except:
            logger.info('No .env found')

        self.client = openai.OpenAI(base_url=uri)
        self.model_id = model_id
        self.messages = messages
        self.total_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.keep_history = keep_history

    @backoff.on_exception(backoff.expo, openai.RateLimitError, max_time=300)
    def __call__(
        self,
        message: str,
        json_mode: bool = False,
        **kwargs
    ) -> str:

        logger.debug(f'Formatted the message')

        if self.keep_history:
            self._add_msg(message, json_mode, 'user')
            response = self._get_response(
                self.messages, json_mode, **kwargs
            )
            self._add_msg(message, json_mode, 'assistant')
            return response

        response = self._get_response(
            self.messages + [{'role': 'user', 'content': message}], json_mode, **kwargs
        )

        return response

    def _get_response(self, messages, json_mode, **kwargs) -> str:
        logger.debug(f'Sending request to the LLM with JSON mode: {json_mode}')
        logger.info(f'Length of the message stream: {len(messages)}')

        if json_mode:
            completions = self.client.chat.completions.create(
                model=self.model_id,
                response_format={'type': 'json_object'},
                messages=messages,
                **kwargs
            )

            if completions.choices[0].finish_reason == 'length':
                raise IOError(f'Reached maximum output length, output format is not reliable. {completions.choices[0].message.content}')

            op = json.loads(completions.choices[0].message.content)

        else:
            completions = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                **kwargs
            )

            op = completions.choices[0].message.content

        logger.info(f'Prompt: {messages[-1]}, output: {op}')
        logger.info(f'Tokens used in generation using {self.model_id}: {completions.usage}')

        self.total_tokens += completions.usage.total_tokens
        self.input_tokens += completions.usage.prompt_tokens
        self.output_tokens += completions.usage.completion_tokens

        return op

    def _add_msg(self, x: str, json_mode: bool = False, role: str = 'user'):

        assert role in ['system', 'assistant', 'user'], 'Role should be one of (user, system, assistant)'

        if json_mode:
            {'role': role, 'content': json.dumps(x)}

        else:
            self.messages.append({
                'role': role,
                'content': x
            })
