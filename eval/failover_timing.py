"""How long does it take to fall back to the second model, under different retry settings?

Uses real OpenAI calls for the timeout and bad-key cases and LiteLLM's mock errors for rate limit and
server error. It prints a table and saves nothing.

    PYTHONPATH=src python eval/failover_timing.py
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litellm import Router  # noqa: E402
from litellm.router import RetryPolicy  # noqa: E402

MSG = [{"role": "user", "content": "say hi"}]


def policy(n: int) -> RetryPolicy:
    return RetryPolicy(TimeoutErrorRetries=n, RateLimitErrorRetries=n, InternalServerErrorRetries=n,
                       BadRequestErrorRetries=0, AuthenticationErrorRetries=0, ContentPolicyViolationErrorRetries=0)


def make(first: dict, retries: int) -> Router:
    return Router(
        model_list=[{"model_name": "m0", "litellm_params": first},
                    {"model_name": "m1", "litellm_params": {"model": "openai/gpt-4o-mini"}}],
        fallbacks=[{"m0": ["m1"]}], num_retries=retries, timeout=30, retry_policy=policy(retries),
        allowed_fails=3, cooldown_time=30)


FAILURES = {
    "timeout (50 ms)": {"model": "openai/gpt-4o", "timeout": 0.05},
    "rate limit": {"model": "openai/gpt-4o", "mock_response": "litellm.RateLimitError"},
    "server error": {"model": "openai/gpt-4o", "mock_response": "litellm.InternalServerError"},
    "bad key": {"model": "openai/gpt-4o", "api_key": "sk-bad"},
}
RETRIES = {"2 retries": 2, "1 retry": 1, "no retries": 0}


async def main() -> None:
    print(f"{'failure':18}" + "".join(f"{name:22}" for name in RETRIES))
    for name, first in FAILURES.items():
        cells = []
        for retries in RETRIES.values():
            router = make(first, retries)
            started = time.time()
            reply = await router.acompletion(model="m0", messages=MSG, max_tokens=5)
            fell_back = "mini" in reply.model
            cells.append(f"{time.time() - started:5.1f}s {'fell back' if fell_back else 'NO FALLBACK':11}")
        print(f"{name:18}" + "".join(f"{c:22}" for c in cells))


if __name__ == "__main__":
    asyncio.run(main())
