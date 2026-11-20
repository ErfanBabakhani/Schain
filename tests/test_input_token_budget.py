import pytest

from schain.llm.provider import InputBudgetLLM, InputTokenBudgetExceeded


class EchoLLM:
    def invoke(self, value, **kwargs):
        return value

    async def ainvoke(self, value, **kwargs):
        return value


def test_input_budget_proxy_accounts_before_provider_call():
    reserved = []
    llm = InputBudgetLLM(EchoLLM(), reserved.append)
    assert llm.invoke("prompt") == "prompt"
    assert reserved == [InputBudgetLLM.estimate_input_tokens(("prompt",), {})]


def test_budget_rejection_prevents_provider_call():
    class Reject:
        def __call__(self, estimate):
            raise InputTokenBudgetExceeded("budget")

    with pytest.raises(InputTokenBudgetExceeded):
        InputBudgetLLM(EchoLLM(), Reject()).invoke("prompt")
