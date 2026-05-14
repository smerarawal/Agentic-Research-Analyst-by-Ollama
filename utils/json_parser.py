import json
import re
import time


def safe_json_parse(text):

    try:
        return json.loads(text)

    except Exception:

        match = re.search(r'\{.*\}', text, re.DOTALL)

        if match:

            return json.loads(match.group())

        raise ValueError("Invalid JSON")


def retry_parser(llm_function, retries=3):

    for attempt in range(retries):

        try:

            result = llm_function()

            return safe_json_parse(result)

        except Exception:

            time.sleep(1)

    raise ValueError("Failed after retries")