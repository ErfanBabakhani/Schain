import pytest

from schain.llm.provider import InputBudgetLLM, InputTokenBudgetExceeded


class EchoLLM:
    def invoke(self, value, **kwargs):
        return value

    async def ainvoke(self, value, **kwargs):
        return value


class CountingTokenizer:
    chat_template = "configured-template"

    def __init__(self):
        self.messages = None
        self.template_kwargs = None

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        self.messages = messages
        self.template_kwargs = kwargs
        assert tokenize and add_generation_prompt
        return list(range(sum(len(message["content"].split()) + 1 for message in messages)))

    def encode(self, value, *, add_special_tokens=False):
        return value.split()


def test_input_budget_proxy_accounts_before_provider_call():
    reserved = []
    tokenizer = CountingTokenizer()
    llm = InputBudgetLLM(EchoLLM(), reserved.append, tokenizer=tokenizer,
                         tokenizer_model="Qwen/test")
    assert llm.invoke("prompt") == "prompt"
    assert reserved == [InputBudgetLLM.estimate_input_tokens(
        ("prompt",), {}, tokenizer=tokenizer,
    )]
    assert tokenizer.messages == [{"role": "user", "content": "prompt"}]


def test_budget_rejection_prevents_provider_call():
    class Reject:
        def __call__(self, estimate):
            raise InputTokenBudgetExceeded("budget")

    with pytest.raises(InputTokenBudgetExceeded):
        InputBudgetLLM(EchoLLM(), Reject(), tokenizer=CountingTokenizer(),
                       tokenizer_model="Qwen/test").invoke("prompt")


def test_request_count_uses_qwen_chat_template_and_system_history():
    tokenizer = CountingTokenizer()
    count = InputBudgetLLM.estimate_input_tokens(
        ("user prompt", [{"role": "assistant", "content": "past reply"}], "system rule"),
        {"response_format": {"type": "json_object"}}, tokenizer=tokenizer,
    )
    assert count > 0
    assert tokenizer.messages[0]["role"] == "system"
    assert any(message["content"] == "past reply" for message in tokenizer.messages)
    assert tokenizer.messages[-1] == {"role": "user", "content": "user prompt"}
    assert all("Requested response format" not in message["content"]
               for message in tokenizer.messages)


def test_request_count_uses_configured_qwen_template_options():
    class ConfiguredEchoLLM(EchoLLM):
        model_params = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}

    tokenizer = CountingTokenizer()
    InputBudgetLLM(
        ConfiguredEchoLLM(), lambda _: None, tokenizer=tokenizer, tokenizer_model="Qwen/test",
    ).count_request_tokens(("prompt",), {})

    assert tokenizer.template_kwargs == {"enable_thinking": False}


def test_batch_input_context_limit_leaves_room_for_configured_output():
    class ContextTokenizer(CountingTokenizer):
        model_max_length = 4096

    class ConfiguredEchoLLM(EchoLLM):
        model_params = {"max_tokens": 2048}

    llm = InputBudgetLLM(
        ConfiguredEchoLLM(), lambda _: None, tokenizer=ContextTokenizer(),
        tokenizer_model="Qwen/test",
    )

    assert llm.input_context_limit_tokens() == 2048


def test_unknown_model_context_length_is_not_treated_as_a_real_limit():
    tokenizer = CountingTokenizer()
    tokenizer.model_max_length = 10**30
    llm = InputBudgetLLM(
        EchoLLM(), lambda _: None, tokenizer=tokenizer, tokenizer_model="Qwen/test",
    )

    assert llm.input_context_limit_tokens() is None
