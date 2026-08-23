"""Dataset schema normalization for AssistiveMind fine-tuning.

The dataset pack ships two schemas:
  - flat:  {"system": ..., "domain": ..., "instruction": ..., "response": ...}
  - chat:  {"domain": ..., "messages": [{"role": ..., "content": ...}, ...]}

Both are normalized to a `messages` list here, then formatted with the
target model's own chat template. That keeps the system prompt a real
training turn instead of metadata bolted onto a hand-built string, and
it means the training script never depends on an `output` key that
neither schema has ever had.
"""

USER_PREFIX = "User: "


def to_messages(example: dict) -> list[dict]:
    if "messages" in example:
        return example["messages"]

    if "instruction" in example and "response" in example:
        messages = []
        system = example.get("system")
        if system:
            messages.append({"role": "system", "content": system})

        instruction = example["instruction"]
        if instruction.startswith(USER_PREFIX):
            instruction = instruction[len(USER_PREFIX):]
        messages.append({"role": "user", "content": instruction})
        messages.append({"role": "assistant", "content": example["response"]})
        return messages

    raise KeyError(
        "Example matches neither the chat schema ('messages') nor the flat "
        f"schema ('instruction'/'response'). Keys found: {sorted(example.keys())}"
    )


def format_example(example: dict, tokenizer) -> str:
    messages = to_messages(example)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
