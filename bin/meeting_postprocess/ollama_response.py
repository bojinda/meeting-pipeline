"""Allowlisted response diagnostics: never retain reasoning or backend errors."""


class GenerationFailure(ValueError):
    def __init__(self, category):
        super().__init__(category)
        self.category = category


def metadata(data, num_predict, thinking):
    data = data if isinstance(data, dict) else {}
    response, trace = data.get("response"), data.get("thinking")
    reason = data.get("done_reason")
    return {"response_char_count": len(response) if isinstance(response, str) else None,
            "done": data.get("done") if type(data.get("done")) is bool else None,
            "thinking_char_count": len(trace) if isinstance(trace, str) else None,
            "done_reason": reason if reason in ("stop", "length", "max_tokens", "load", "unload", "error") else "other" if reason is not None else None,
            "eval_count": data.get("eval_count") if type(data.get("eval_count")) is int else None,
            "prompt_eval_count": data.get("prompt_eval_count") if type(data.get("prompt_eval_count")) is int else None,
            "num_predict": num_predict, "requested_thinking": "default" if thinking is None else thinking,
            "empty_final_answer": isinstance(response, str) and not response.strip(),
            "thinking_only": isinstance(response, str) and not response.strip() and isinstance(trace, str) and bool(trace.strip())}


def classify(data, info):
    if not isinstance(data, dict) or "error" in data or not isinstance(data.get("response"), str):
        return "ollama_invalid_response"
    if info["done_reason"] == "error":
        return "ollama_error_termination"
    if data.get("done") is not True:
        return "ollama_incomplete_response"
    if (info["done_reason"] in {"length", "max_tokens"} or
            info["num_predict"] is not None and info["eval_count"] is not None and info["eval_count"] >= info["num_predict"]):
        return "ollama_generation_limit"
    if info["thinking_only"]:
        return "ollama_thinking_only_answer"
    if info["empty_final_answer"]:
        return "ollama_empty_final_answer"
    if "<think>" in data["response"] or "</think>" in data["response"]:
        return "ollama_unseparated_thinking"
    return None
