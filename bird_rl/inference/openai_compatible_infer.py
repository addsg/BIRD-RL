#!/usr/bin/env python3
"""Run batch inference against an OpenAI-compatible Chat Completions API."""

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm


RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


def request_completion(
    session: requests.Session,
    url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    retries: int,
    timeout: float,
) -> tuple[str, dict]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    delay = 2.0
    for attempt in range(retries + 1):
        try:
            response = session.post(url, headers=headers, json=payload, timeout=timeout)
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"].get("content") or ""
                return content, result.get("usage", {})
            if response.status_code not in RETRYABLE_STATUS or attempt == retries:
                raise RuntimeError(
                    f"HTTP {response.status_code}: {response.text[:1000]}"
                )
        except (requests.RequestException, ValueError) as exc:
            if attempt == retries:
                raise RuntimeError(f"API request failed: {exc}") from exc

        time.sleep(delay)
        delay = min(delay * 2, 30)

    raise RuntimeError("API request failed after retries")


def run_inference(args: argparse.Namespace) -> None:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(
            f"Environment variable {args.api_key_env} is not set. "
            f"Use: read -s {args.api_key_env}; export {args.api_key_env}"
        )

    base_url = args.base_url.rstrip("/")
    url = f"{base_url}/chat/completions"
    with open(args.prompt_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(data)} prompts")
    print(f"Endpoint: {base_url}")
    print(f"Model: {args.model}")
    print(f"Concurrency: {args.num_threads}")

    thread_local = threading.local()

    def process(index: int) -> tuple[int, str, dict, str | None]:
        if not hasattr(thread_local, "session"):
            thread_local.session = requests.Session()
        item = data[index]
        try:
            content, usage = request_completion(
                session=thread_local.session,
                url=url,
                api_key=api_key,
                model=args.model,
                system_prompt=item.get("system_prompt", ""),
                user_prompt=item.get("prompt", ""),
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                retries=args.retries,
                timeout=args.timeout,
            )
            return index, content, usage, None
        except Exception as exc:
            return index, "", {}, str(exc)

    failures = 0
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = [executor.submit(process, i) for i in range(len(data))]
        for future in tqdm(as_completed(futures), total=len(futures), desc="API inference"):
            index, content, usage, error = future.result()
            data[index]["raw_response"] = content
            data[index]["api_model"] = args.model
            data[index]["api_usage"] = usage
            if error:
                failures += 1
                data[index]["api_error"] = error
            for key in total_usage:
                total_usage[key] += int(usage.get(key, 0) or 0)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(data)} responses to {output_path}")
    print(f"Failures: {failures}")
    print(f"Token usage: {total_usage}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible batch inference for critic prompts"
    )
    parser.add_argument("--prompt_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL"))
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--num_threads", type=int, default=2)
    parser.add_argument("--max_tokens", type=int, default=3000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    if not args.base_url:
        parser.error("--base_url or OPENAI_BASE_URL is required")
    if not args.model:
        parser.error("--model or OPENAI_MODEL is required")
    run_inference(args)


if __name__ == "__main__":
    main()
