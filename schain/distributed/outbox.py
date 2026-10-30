"""Crash-safe Neo4j dispatcher for PostgreSQL graph write intents."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from ..config import neo4j_settings
from ..kg.writer import facts_to_graph_rows, merge_nodes, merge_relationships
from .models import StaleLeaseError

logger = logging.getLogger(__name__)


class Neo4jOutboxDispatcher:
    def __init__(
        self,
        repository: Any,
        driver: Any,
        config: dict[str, Any],
        ontology: dict[str, Any],
        *,
        experiment_id: str,
        worker_id: str,
        lease_seconds: int = 120,
    ) -> None:
        if driver is None:
            raise ValueError("Save ON requires an initialized Neo4j driver")
        self.repository = repository
        self.driver = driver
        self.config = config
        self.ontology = ontology
        self.experiment_id = experiment_id
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds

    def drain(self, *, max_items: int = 0) -> int:
        """Replay pending intents; zero means drain until none is immediately ready."""
        written = 0
        while max_items <= 0 or written < max_items:
            lease = self.repository.claim_outbox(
                self.experiment_id, self.worker_id, lease_seconds=self.lease_seconds,
            )
            if lease is None:
                break
            try:
                facts = pd.DataFrame(lease.payload.get("facts") or [])
                nodes, relationships = facts_to_graph_rows(facts)
                settings = neo4j_settings(self.config)
                node_count = merge_nodes(
                    self.driver, settings["database"], nodes,
                    self.ontology["entity_types"], settings["batch_size"],
                )
                relationship_count = merge_relationships(
                    self.driver, settings["database"], relationships,
                    self.ontology["relation_types"], settings["batch_size"],
                )
                # If the process dies before this ack, replay is safe because both
                # writers use canonical MERGE keys and set-union provenance.
                self.repository.acknowledge_outbox(
                    lease, nodes=node_count, relationships=relationship_count,
                )
                written += 1
            except Exception as error:  # noqa: BLE001 - retained durably for retry
                try:
                    self.repository.retry_outbox(lease, error)
                except StaleLeaseError:
                    # Another dispatcher already fenced/reclaimed it. The Neo4j MERGE
                    # may have happened, so the authoritative owner will safely replay.
                    logger.warning("outbox retry fenced after graph attempt")
                logger.warning("Neo4j outbox write retained for retry: %s", type(error).__name__)
                break
        return written
