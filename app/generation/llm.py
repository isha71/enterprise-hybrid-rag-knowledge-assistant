import httpx
from openai import OpenAI

from app.core.config import settings
from app.core.logging import logger


class LLMClient:
    """LLM client supporting Ollama (local) and OpenAI (hosted) providers."""

    def __init__(self):
        self._openai_client: OpenAI | None = None

    @property
    def _openai(self) -> OpenAI:
        if self._openai_client is None:
            if not settings.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY is not set. Please add it to your .env file."
                )
            self._openai_client = OpenAI(api_key=settings.openai_api_key)
        return self._openai_client

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response using the configured LLM provider."""
        provider = settings.llm_provider.lower()
        logger.info(f"Calling LLM: provider={provider}, model={settings.llm_model}")

        if provider == "ollama":
            return self._generate_ollama(system_prompt, user_prompt)
        elif provider == "openai":
            return self._generate_openai(system_prompt, user_prompt)
        else:
            raise ValueError(
                f"Unsupported LLM_PROVIDER: '{provider}'. Use 'ollama' or 'openai'."
            )

    def _generate_ollama(self, system_prompt: str, user_prompt: str) -> str:
        """Call the native Ollama /api/chat endpoint."""
        url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
        payload = {
            "model": settings.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "think": False,
            "stream": False,
            "options": {
                "temperature": 0.1,
            },
        }

        try:
            resp = httpx.post(url, json=payload, timeout=120.0)
            resp.raise_for_status()
        except httpx.ConnectError as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {settings.ollama_base_url}. "
                f"Ensure Ollama is running: {e}"
            ) from e
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Ollama returned HTTP {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.TimeoutException as e:
            raise TimeoutError(
                f"Ollama request timed out after 120s: {e}"
            ) from e

        data = resp.json()
        message = data.get("message")
        if not message or "content" not in message:
            raise RuntimeError(
                f"Unexpected Ollama response structure: {data}"
            )

        answer = message["content"]
        logger.info(f"LLM response received ({len(answer)} chars)")
        return answer

    def _generate_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Call the OpenAI chat completions API."""
        response = self._openai.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1024,
        )

        answer = response.choices[0].message.content or ""
        logger.info(f"LLM response received ({len(answer)} chars)")
        return answer


llm_client = LLMClient()
