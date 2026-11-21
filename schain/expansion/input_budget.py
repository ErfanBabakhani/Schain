"""Durable request accounting shared by EXPAND and the selection benchmark."""

from contextlib import closing, nullcontext
import json
from pathlib import Path
import sqlite3
import threading

from .. import state
from ..config import REPO_ROOT
from ..llm.provider import InputBudgetLLM, InputTokenBudgetExceeded


def budgeted_llm(config, connection, run_id, llm, tokenizer, tokenizer_model, *, strict=False):
    key = f"input_budget:{run_id}"
    selection = config["expansion"]["selection"]
    limit = int(selection["total_input_tokens"])
    ledger = state.run_value(connection, key) or {
        "spent_estimate": 0, "reserved_estimate": 0, "limit": limit,
        "requests": 0, "accounting_version": "qwen_tokenizer_v1",
    }
    if ledger.get("accounting_version") != "qwen_tokenizer_v1":
        raise ValueError("input-token ledger uses character estimates; start a fresh EXPAND state")
    if int(ledger["limit"]) != limit:
        raise ValueError("input-token budget changed; start a fresh EXPAND state")
    if ledger.get("tokenizer_model") not in (None, tokenizer_model):
        raise ValueError("input-token tokenizer changed; start a fresh EXPAND state")
    ledger.update(tokenizer_model=tokenizer_model, reserved_estimate=0)
    state.set_run_value(connection, key, ledger)
    if llm is None or getattr(llm, "_schain_input_budget", False):
        return llm

    owner = threading.get_ident()
    db_path = connection.execute("PRAGMA database_list").fetchone()[2]
    lock = threading.Lock()
    log_dir = Path(config.get("storage", {}).get("logs", REPO_ROOT / "logs"))

    def access():
        if threading.get_ident() == owner or not db_path:
            return nullcontext(connection)
        # Extraction can invoke the model on the persistent asyncio thread.
        return closing(sqlite3.connect(db_path, timeout=30))

    def log(entry):
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / f"input_token_requests_{run_id}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "run_id": run_id, "accounting_source": "configured tokenizer chat-template token count",
                "tokenizer_model": tokenizer_model, **entry,
            }) + "\n")

    def reserve(tokens):
        with lock, access() as db:
            budget = state.run_value(db, key)
            spent = int(budget["spent_estimate"])
            remaining = limit - spent
            if strict and budget.get("first_rejection"):
                raise InputTokenBudgetExceeded("a request already reached the input-token limit")
            reserved = 0 if strict else int(budget.get("reserved_estimate", 0))
            entry = {
                "request_index": budget.get("requests", 0) + 1, "input_tokens": int(tokens),
                "spent_before": spent,
            }
            if tokens + reserved > remaining:
                budget["rejected_requests"] = int(budget.get("rejected_requests", 0)) + 1
                budget.setdefault("first_rejection", {
                    "request_input_tokens": int(tokens), "remaining_input_tokens": remaining,
                })
                state.set_run_value(db, key, budget)
                log({**entry, "status": "rejected_by_budget"})
                raise InputTokenBudgetExceeded(
                    f"input-token budget exhausted: {spent} + {tokens} > {limit}"
                )
            budget["spent_estimate"] = spent + int(tokens)
            budget["requests"] = entry["request_index"]
            if strict:
                phase = getattr(wrapped, "benchmark_phase", "unassigned")
                bucket = budget.setdefault("phases", {}).setdefault(phase, {
                    "input_tokens": 0, "requests": 0,
                })
                bucket["input_tokens"] += int(tokens)
                bucket["requests"] += 1
                entry["phase"] = phase
            state.set_run_value(db, key, budget)
            log({**entry, "spent_after": budget["spent_estimate"]})

    def output_usage(response):
        if isinstance(response, dict):
            usage = response.get("usage")
        else:
            usage = getattr(response, "usage", None)
            metadata = getattr(response, "response_metadata", None)
            if usage is None and isinstance(metadata, dict):
                usage = metadata.get("token_usage") or metadata.get("usage")
        if usage is None:
            return
        for field in ("response_tokens", "completion_tokens", "output_tokens"):
            count = usage.get(field) if isinstance(usage, dict) else getattr(usage, field, None)
            if isinstance(count, int) and count >= 0:
                with lock, access() as db:
                    budget = state.run_value(db, key)
                    budget["output_tokens"] = budget.get("output_tokens", 0) + count
                    budget["responses_with_output_usage"] = budget.get("responses_with_output_usage", 0) + 1
                    state.set_run_value(db, key, budget)
                break

    wrapped = InputBudgetLLM(
        llm, reserve, tokenizer=tokenizer, tokenizer_model=tokenizer_model,
        strict_template=strict, record_response=output_usage if strict else None,
    )
    return wrapped
