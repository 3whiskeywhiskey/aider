import hashlib, os
import json

import backoff
import httpx
import openai

# from diskcache import Cache
from openai import APIConnectionError, InternalServerError, RateLimitError

from aider.dump import dump  # noqa: F401

CACHE_PATH = "~/.aider.send.cache.v1"
CACHE = None
# CACHE = Cache(CACHE_PATH)


@backoff.on_exception(
    backoff.expo,
    (
        InternalServerError,
        RateLimitError,
        APIConnectionError,
        httpx.ConnectError,
    ),
    max_tries=10,
    on_backoff=lambda details: print(
        f"{details.get('exception','Exception')}\nRetry in {details['wait']:.1f} seconds."
    ),
)
def send_with_retries(client, model_name, messages, functions, stream, log_file_path=None):
    if not client:
        raise ValueError("No openai client provided")

    kwargs = dict(
        model=model_name,
        messages=messages,
        temperature=0,
        stream=stream,
    )
    if functions is not None:
        kwargs["functions"] = functions

    key = json.dumps(kwargs, sort_keys=True).encode()

    # Generate SHA1 hash of kwargs and append it to chat_completion_call_hashes
    hash_object = hashlib.sha1(key)

    if not stream and CACHE is not None and key in CACHE:
        return hash_object, CACHE[key]

    res = client.chat.completions.create(**kwargs)

    if log_file_path is not None:
        with open(os.path.expanduser(log_file_path), "a") as log_file:
            log_file.write(f"Prompt: {json.dumps(kwargs, sort_keys=True, ensure_ascii=False)}\n")
            log_file.write(f"Response: {json.dumps(res, ensure_ascii=False)}\n\n")

    if not stream and CACHE is not None:
        CACHE[key] = res

    return hash_object, res


def simple_send_with_retries(client, model_name, messages):
    log_file_path = "chat_log.txt"  # Default log file path
    try:
        _hash, response = send_with_retries(
            client=client,
            model_name=model_name,
            messages=messages,
            functions=None,
            stream=False,
            log_file_path=log_file_path,
        )
        return response.choices[0].message.content
    except (AttributeError, openai.BadRequestError):
        return
