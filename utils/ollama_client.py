import ollama

DEFAULT_MODEL = "llama3"

SYSTEM_PROMPT = (
    "You are a rigorous research analyst. "
    "Be specific, cite facts, avoid filler phrases like "
    "'it is important to note' or 'in conclusion'. "
    "Never pad responses with generic statements."
)


def query_ollama(prompt, model=DEFAULT_MODEL):

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]
