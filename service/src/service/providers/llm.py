from collections.abc import AsyncIterator

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from service.providers.base import ChatPrompt


class OpenAILLM:
    def __init__(self, api_key: str, model: str):
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def stream(self, prompt: ChatPrompt, history: list) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=self._model, messages=self._messages(prompt, history), stream=True
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    async def complete(self, prompt: ChatPrompt, history: list) -> str:
        return "".join([t async for t in self.stream(prompt, history)])

    def _messages(
        self, prompt: ChatPrompt, history: list
    ) -> list[ChatCompletionMessageParam]:
        msgs: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": prompt.system}
        ]
        msgs.extend({"role": r, "content": c} for r, c in history)
        if prompt.context:
            ctx = "\n\n".join(prompt.context)
            msgs.append({"role": "user", "content": f"Context:\n{ctx}"})
        msgs.append({"role": "user", "content": prompt.user})
        return msgs
