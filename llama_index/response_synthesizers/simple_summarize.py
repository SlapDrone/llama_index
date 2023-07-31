from typing import Any, Generator, Optional, Sequence, cast

from llama_index.indices.service_context import ServiceContext
from llama_index.llms.base import ChatMessage, MessageRole
from llama_index.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT
from llama_index.prompts.prompts import QuestionAnswerPrompt
from llama_index.response_synthesizers.base import (
    BaseSynthesizer,
    convert_llm_output_to_legacy,
)
from llama_index.types import RESPONSE_TEXT_TYPE


class SimpleSummarize(BaseSynthesizer):
    def __init__(
        self,
        text_qa_template: Optional[QuestionAnswerPrompt] = None,
        service_context: Optional[ServiceContext] = None,
        streaming: bool = False,
    ) -> None:
        super().__init__(service_context=service_context, streaming=streaming)
        self._text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT

    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        text_qa_template = self._text_qa_template.partial_format(query_str=query_str)
        truncated_chunks = self._service_context.prompt_helper.truncate(
            prompt=text_qa_template,
            text_chunks=text_chunks,
        )
        node_text = "\n".join(truncated_chunks)

        response: RESPONSE_TEXT_TYPE
        prompt = text_qa_template.format(context_str=node_text)
        if not self._streaming:
            response = await self._service_context.llm.achat(
                [ChatMessage(role=MessageRole.USER, content=prompt)]
            )
        else:
            response = self._service_context.llm.astream_chat(
                [ChatMessage(role=MessageRole.USER, content=prompt)]
            )

        if isinstance(response, str):
            response = response or "Empty Response"
        else:
            response = cast(Generator, response)

        return response

    def get_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        **kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        text_qa_template = self._text_qa_template.partial_format(query_str=query_str)
        truncated_chunks = self._service_context.prompt_helper.truncate(
            prompt=text_qa_template,
            text_chunks=text_chunks,
        )
        node_text = "\n".join(truncated_chunks)

        response: RESPONSE_TEXT_TYPE
        prompt = text_qa_template.format(context_str=node_text)

        if not self._streaming:
            response = self._service_context.llm.chat(
                [ChatMessage(role=MessageRole.USER, content=prompt)]
            )
        else:
            response = self._service_context.llm.stream_chat(
                [ChatMessage(role=MessageRole.USER, content=prompt)]
            )

        if isinstance(response, str):
            response = response or "Empty Response"
        else:
            response = cast(Generator, response)

        return response
