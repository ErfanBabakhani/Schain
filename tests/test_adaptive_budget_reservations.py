from schain import state
from schain.expansion.loop import (
    _affordable_action_indices,
    _finish_adaptive_reservations,
    _settle_adaptive_cost,
)


def test_batch_selection_fits_combined_cost_and_keeps_nonconsecutive_snippets():
    choices = {2: "snippet", 5: "snippet", 8: "snippet"}
    details = {
        2: ({}, {"snippet": 60}),
        5: ({}, {"snippet": 41}),
        8: ({}, {"snippet": 40}),
    }

    selected = _affordable_action_indices(
        [2, 5, 8], choices, details, remaining_tokens=100, max_count=10,
    )

    assert selected == [2, 8]
    assert sum(details[index][1][choices[index]] for index in selected) == 100


def test_reservations_release_after_an_unprocessed_batch_and_track_actual_queue_cost(connection):
    run_id = "reservation-test"
    state.set_run_value(connection, f"input_budget:{run_id}", {
        "spent_estimate": 125, "reserved_estimate": 90, "limit": 200,
    })
    state.set_run_value(connection, f"adaptive_queue_spent:{run_id}", 100)
    state.set_run_value(connection, f"adaptive_virtual_price:{run_id}", .4)
    decisions = [
        {
            "estimated_cost": 50, "reserved_cost": 50, "batch_input_tokens": 25,
            "decision": {"queue_before": .4},
        },
        {
            "estimated_cost": 40, "reserved_cost": 40, "batch_input_tokens": 0,
            "decision": {"queue_before": .4},
        },
    ]

    _finish_adaptive_reservations(connection, run_id, decisions, {
        "target_tokens_per_candidate": 20, "cost_reference_tokens": 100,
    })

    budget = state.run_value(connection, f"input_budget:{run_id}")
    assert budget["reserved_estimate"] == 0
    assert state.run_value(connection, f"adaptive_virtual_price:{run_id}") == .45
    assert state.run_value(connection, f"adaptive_virtual_queue_history:{run_id}") == [{
        "queue_before": .4, "queue_after": .45,
        "input_tokens": 25, "candidate_count": 1,
    }]
    assert all(decision["cost_settled"] for decision in decisions[:1])


def test_settled_adaptive_cost_is_not_charged_twice(connection):
    run_id = "settled-cost-test"
    state.set_run_value(connection, f"input_budget:{run_id}", {
        "spent_estimate": 80, "reserved_estimate": 40, "limit": 200,
    })
    state.set_run_value(connection, f"adaptive_queue_spent:{run_id}", 50)
    state.set_run_value(connection, f"adaptive_virtual_price:{run_id}", .4)
    decision = {
        "reserved_cost": 40,
        "decision": {"queue_before": .4},
    }
    selection = {"target_tokens_per_candidate": 20, "cost_reference_tokens": 100}

    _settle_adaptive_cost(
        connection, run_id, decision, selection, actual_cost=25,
    )
    _settle_adaptive_cost(
        connection, run_id, decision, selection, actual_cost=25,
    )

    assert state.run_value(connection, f"adaptive_virtual_price:{run_id}") == .45
    assert state.run_value(connection, f"input_budget:{run_id}")["reserved_estimate"] == 0
    assert len(state.run_value(connection, f"adaptive_virtual_queue_history:{run_id}")) == 1
