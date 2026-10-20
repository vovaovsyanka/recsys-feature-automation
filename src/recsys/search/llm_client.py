"""Клиент языковой модели для эксперимента с ИИ-агентом.

Порядок источников: облако Ollama с ключом OLLAMA_API_KEY, затем с ключом OLLAMA_API_KEY2 (если у первого кончились
бесплатные кредиты или лимит), затем локальный сервер с OpenAI-совместимым API (LM Studio, адрес и модель задаются
переменными RECSYS_LOCAL_URL и RECSYS_LOCAL_MODEL). Ключи нигде не печатаются и не записываются в логи.
"""
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

CLOUD_URL = "https://ollama.com/api/chat"
LOCAL_URL = os.environ.get("RECSYS_LOCAL_URL", "http://localhost:1234/v1")
LOCAL_MODEL = os.environ.get("RECSYS_LOCAL_MODEL", "openai/gpt-oss-20b")

# коды ответа, при которых ключ считается исчерпанным и берётся следующий источник
_EXHAUSTED = {401, 402, 403, 429}


@dataclass
class Reply:
    text: str
    tokens_in: int
    tokens_out: int
    seconds: float
    source: str          # cloud-1, cloud-2 или local
    model: str


class LLMClient:
    def __init__(self, model: str = "gpt-oss:120b", temperature: float = 0.7, seed: int = 0, timeout: int = 300):
        self.model, self.temperature, self.seed, self.timeout = model, temperature, seed, timeout
        keys = [os.environ.get("OLLAMA_API_KEY"), os.environ.get("OLLAMA_API_KEY2")]
        self.sources = [(f"cloud-{i + 1}", k) for i, k in enumerate(keys) if k] + [("local", None)]
        self.dead: set[str] = set()
        self.usage = {"calls": 0, "tokens_in": 0, "tokens_out": 0, "seconds": 0.0, "by_source": {}}

    def _post(self, url: str, body: dict, key: str | None) -> dict:
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _call(self, source: str, key: str | None, messages: list[dict], max_tokens: int) -> Reply:
        t = time.time()
        if source == "local":
            body = {"model": LOCAL_MODEL, "messages": messages, "temperature": self.temperature,
                    "max_tokens": max_tokens, "seed": self.seed}
            out = self._post(LOCAL_URL.rstrip("/") + "/chat/completions", body, None)
            text = out["choices"][0]["message"]["content"]
            tin, tout, model = out.get("usage", {}).get("prompt_tokens", 0), out.get("usage", {}).get("completion_tokens", 0), LOCAL_MODEL
        else:
            body = {"model": self.model, "messages": messages, "stream": False,
                    "options": {"temperature": self.temperature, "seed": self.seed, "num_predict": max_tokens}}
            out = self._post(CLOUD_URL, body, key)
            text = out["message"]["content"]
            tin, tout, model = out.get("prompt_eval_count", 0), out.get("eval_count", 0), self.model
        return Reply(text, tin, tout, time.time() - t, source, model)

    def chat(self, system: str, user: str, max_tokens: int = 4000, retries: int = 3) -> Reply:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        last_error = None
        for source, key in self.sources:
            if source in self.dead:
                continue
            for attempt in range(retries):
                try:
                    reply = self._call(source, key, messages, max_tokens)
                    self._count(reply)
                    return reply
                except urllib.error.HTTPError as e:
                    last_error = f"{source}: HTTP {e.code}"
                    if e.code in _EXHAUSTED:
                        self.dead.add(source)
                        print(f"  [llm] источник {source} недоступен (HTTP {e.code}), переключаюсь на следующий")
                        break
                    time.sleep(2 ** attempt * 3)
                except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                    last_error = f"{source}: {type(e).__name__}"
                    if source == "local":
                        break                    # локальный сервер не запущен - повторять бессмысленно
                    time.sleep(2 ** attempt * 3)
        raise RuntimeError(f"ни один источник LLM не ответил (последняя ошибка: {last_error}). "
                           "Проверьте OLLAMA_API_KEY/OLLAMA_API_KEY2 или запустите сервер LM Studio.")

    def _count(self, r: Reply) -> None:
        u = self.usage
        u["calls"] += 1
        u["tokens_in"] += r.tokens_in
        u["tokens_out"] += r.tokens_out
        u["seconds"] += r.seconds
        u["by_source"][r.source] = u["by_source"].get(r.source, 0) + 1
