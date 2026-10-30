"""Checkpoint state: document status flow, deduplication, frontier and caches."""

from __future__ import annotations

import pytest

from schain import state


def document(doc_id="d1", content_hash="h1", url="https://x.test/1"):
    return {"doc_id": doc_id, "content_hash": content_hash,
            "source_url": url, "source_path": "p"}


def test_document_is_registered_once(connection):
    assert state.register_document(connection, document()) == ("d1", True)
    assert state.register_document(connection, document()) == ("d1", False)


def test_duplicate_content_hash_maps_to_the_first_document(connection):
    state.register_document(connection, document())
    doc_id, is_new = state.register_document(
        connection, document(doc_id="d2", url="https://x.test/other")
    )
    assert (doc_id, is_new) == ("d1", False)


def test_duplicate_normalized_url_maps_to_the_first_document(connection):
    state.register_document(connection, document())
    doc_id, is_new = state.register_document(
        connection, document(doc_id="d3", content_hash="different")
    )
    assert (doc_id, is_new) == ("d1", False)


def test_status_flow_and_attempts(connection):
    state.register_document(connection, document())
    for status in (state.PROCESSING, state.EXTRACTED, state.RESOLVED, state.COMMITTED):
        state.set_document_status(connection, "d1", status)
        assert state.document_status(connection, "d1") == status
    assert state.document_counts(connection) == {state.COMMITTED: 1}


def test_unknown_status_is_rejected(connection):
    state.register_document(connection, document())
    with pytest.raises(ValueError, match="unknown document status"):
        state.set_document_status(connection, "d1", "NOT_A_STATUS")


def test_documents_with_status_filters(connection):
    state.register_document(connection, document())
    state.register_document(connection, document("d2", "h2", "https://x.test/2"))
    state.set_document_status(connection, "d2", state.FAILED, error="boom")
    failed = state.documents_with_status(connection, [state.FAILED])
    assert [row["doc_id"] for row in failed] == ["d2"]
    assert failed[0]["error"] == "boom"


def test_reset_clears_everything(ingest_config, connection):
    from pathlib import Path

    state.register_document(connection, document())
    path = Path(ingest_config["storage"]["state_db"])
    connection.close()
    state.reset(path)
    fresh = state.connect(path)
    assert state.document_counts(fresh) == {}
    fresh.close()


# --- frontier ----------------------------------------------------------------------


def test_frontier_deduplicates_by_canonical_id(connection):
    assert state.enqueue_entity(connection, "BMW", 0) is True
    assert state.enqueue_entity(connection, "BMW AG", 1) is False
    assert state.frontier_tier_counts(connection) == {0: 1}


def test_frontier_serves_lowest_tier_first(connection):
    state.enqueue_entity(connection, "CATL", 1)
    state.enqueue_entity(connection, "BMW", 0)
    assert state.next_frontier_entity(connection, max_tier=3)["entity_name"] == "BMW"


def test_frontier_respects_the_tier_boundary(connection):
    state.enqueue_entity(connection, "Deep Supplier", 4)
    assert state.next_frontier_entity(connection, max_tier=3) is None


def test_claiming_marks_processing_and_resume_requeues_it(connection):
    state.enqueue_entity(connection, "BMW", 0)
    row = state.next_frontier_entity(connection, max_tier=3)
    assert state.frontier_counts(connection) == {state.PROCESSING: 1}
    assert row["attempts"] == 1  # attempts incremented when claimed

    assert state.requeue_stale_processing(connection) == 1
    assert state.frontier_counts(connection) == {state.PENDING: 1}
    # a second claim increments attempts, so retries are visible
    assert state.next_frontier_entity(connection, max_tier=3)["attempts"] == 2


def test_done_entity_is_not_served_again(connection):
    state.enqueue_entity(connection, "BMW", 0)
    row = state.next_frontier_entity(connection, max_tier=3)
    state.set_frontier_status(connection, row["entity_key"], state.DONE)
    assert state.next_frontier_entity(connection, max_tier=3) is None


# --- searches and the resolution cache ---------------------------------------------


def test_completed_search_is_not_repeated(connection):
    sid = state.search_id("company:bmw", 0, "wikipedia", '"BMW" suppliers')
    assert state.search_completed(connection, sid) is False
    state.record_search(connection, sid, "company:bmw", 0, "wikipedia",
                        '"BMW" suppliers', state.DONE, 12)
    assert state.search_completed(connection, sid) is True


def test_failed_search_is_retried(connection):
    sid = state.search_id("company:bmw", 0, "gdelt", '"BMW" suppliers')
    state.record_search(connection, sid, "company:bmw", 0, "gdelt",
                        '"BMW" suppliers', state.FAILED, 0, error="timeout")
    assert state.search_completed(connection, sid) is False


def test_search_id_is_stable_and_distinct():
    a = state.search_id("company:bmw", 0, "wikipedia", "q")
    assert a == state.search_id("company:bmw", 0, "wikipedia", "q")
    assert a != state.search_id("company:bmw", 1, "wikipedia", "q")
    assert a != state.search_id("company:bmw", 0, "gdelt", "q")


def test_resolution_cache_round_trip(connection):
    state.save_resolution(connection, [
        {"label": "Company", "mention": "CATL", "canonical_id": "company:catl",
         "canonical_name": "CATL", "method": "llm"},
    ])
    cache = state.load_resolution_cache(connection)
    assert cache["company:catl"]["canonical_name"] == "CATL"
    # saving the same mention again updates rather than duplicates
    state.save_resolution(connection, [
        {"label": "Company", "mention": "catl", "canonical_id": "company:new",
         "canonical_name": "New", "method": "llm"},
    ])
    assert len(state.load_resolution_cache(connection)) == 1


def test_documents_are_traceable_to_their_frontier_entity(connection):
    state.register_document(connection, document(), origin_entity="company:bmw", tier=0)
    assert state.document_ids_for_entity(connection, "company:bmw") == ["d1"]
    assert state.document_ids_for_entity(connection, "company:other") == []
