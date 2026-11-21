"""Contextual document selection primitives for EXPAND."""

from __future__ import annotations

import math
import logging
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

import numpy as np
from ..names import normalization_key

logger = logging.getLogger(__name__)

FEATURE_VERSION = "schain-adaptive-v1"
FEATURE_NAMES = (
    "bias", "supplier_relevance", "irrelevant_content", "target_match",
    "official_source", "search_rank", "document_novelty", "entity_novelty",
)
ACTIONS = ("snippet", "full_text", "both")
_EMBEDDER = None
_NER = None
_EMBEDDER_MODEL = None
_NER_MODEL = None
_EMBEDDING_CACHE: dict[tuple[str, str], np.ndarray] = {}
_ORG_CACHE: dict[tuple[str, str], set[str]] = {}
_UNKNOWN_OFFICIAL_HOSTS: set[str] = set()


def context_vector(*, supplier_relevance: float, irrelevant_content: float,
                   target_match: bool, official_source: bool, search_rank: int,
                   document_novelty: float, entity_novelty: float) -> np.ndarray:
    """Build the specification's ordered, bounded eight-feature context."""
    values = np.asarray([
        1.0, supplier_relevance, irrelevant_content, float(target_match),
        float(official_source), 1.0 / max(1, int(search_rank)),
        document_novelty, entity_novelty,
    ], dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("adaptive context features must be finite values in [0, 1]")
    return values


def normalize_reward(raw_reward: float, clip_max: float = 5.0) -> float:
    """Clip and normalize the validated graph-gain proxy to the model's [0, 1] range."""
    if not math.isfinite(raw_reward) or raw_reward < 0 or not math.isfinite(clip_max) or clip_max <= 0:
        raise ValueError("reward must be nonnegative and clip_max must be positive")
    return min(raw_reward, clip_max) / clip_max


@dataclass
class LinUCB:
    """Three independent ridge-regression rows shared across EXPAND entities."""

    ridge_lambda: float = 1.0
    alpha: float = 0.5

    def __post_init__(self):
        if self.ridge_lambda <= 0 or self.alpha < 0:
            raise ValueError("ridge_lambda must be positive and alpha nonnegative")
        self.A = np.repeat(np.eye(8)[None, :, :] * self.ridge_lambda, 3, axis=0)
        self.b = np.zeros((3, 8), dtype=float)
        self.W = np.zeros((3, 8), dtype=float)
        self.counts = {action: 0 for action in ACTIONS}

    def scores(self, contexts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        contexts = np.asarray(contexts, dtype=float)
        if contexts.ndim != 2 or contexts.shape[1] != 8:
            raise ValueError("contexts must have shape (n, 8)")
        means = contexts @ self.W.T
        uncertainty = np.empty_like(means)
        for index in range(3):
            solved = np.linalg.solve(self.A[index], contexts.T)
            variance = np.sum(contexts.T * solved, axis=0)
            uncertainty[:, index] = np.sqrt(np.maximum(variance, 0.0))
        return means, means + self.alpha * uncertainty

    def update(self, action: str, context: np.ndarray, reward: float) -> None:
        if action not in ACTIONS:
            raise ValueError(f"unknown learned action: {action}")
        if not math.isfinite(reward) or reward < 0:
            raise ValueError("reward must be finite and nonnegative")
        index = ACTIONS.index(action)
        x = np.asarray(context, dtype=float)
        if x.shape != (8,) or not np.isfinite(x).all():
            raise ValueError("context must be a finite 8-vector")
        self.A[index] += np.outer(x, x)
        self.b[index] += reward * x
        self.W[index] = np.linalg.solve(self.A[index], self.b[index])
        self.counts[action] += 1

    def to_dict(self) -> dict:
        return {"feature_version": FEATURE_VERSION, "actions": list(ACTIONS),
                "ridge_lambda": self.ridge_lambda, "alpha": self.alpha,
                "A": self.A.tolist(), "b": self.b.tolist(),
                "W": self.W.tolist(), "counts": dict(self.counts)}

    @classmethod
    def from_dict(cls, data: dict) -> "LinUCB":
        if data.get("feature_version") != FEATURE_VERSION or tuple(data.get("actions", ())) != ACTIONS:
            raise ValueError("incompatible adaptive selection checkpoint")
        model = cls(float(data["ridge_lambda"]), float(data["alpha"]))
        model.A = np.asarray(data["A"], dtype=float)
        model.b = np.asarray(data["b"], dtype=float)
        model.W = np.asarray(data["W"], dtype=float)
        model.counts = {key: int(data["counts"].get(key, 0)) for key in ACTIONS}
        if model.A.shape != (3, 8, 8) or model.b.shape != (3, 8) or model.W.shape != (3, 8):
            raise ValueError("invalid adaptive selection checkpoint shapes")
        return model


def choose_action(model: LinUCB, context: np.ndarray, *, estimated_costs: dict[str, float],
                  virtual_price: float, cost_reference: float = 1000.0,
                  priors: dict[str, float] | None = None, warmup: bool = False,
                  temperature: float = 1.0, reward_weight: float = 1.0,
                  rng=None) -> tuple[str, dict]:
    """Choose skip or a feasible processing arm using prior-softmax warmup / greedy score."""
    if virtual_price < 0 or cost_reference <= 0 or temperature <= 0 or reward_weight < 0:
        raise ValueError("invalid adaptive selection cost or temperature")
    _, upper = model.scores(np.asarray(context, dtype=float).reshape(1, 8))
    scores = {
        action: reward_weight * float(upper[0, i])
        - virtual_price * float(estimated_costs[action]) / cost_reference
        for i, action in enumerate(ACTIONS) if action in estimated_costs
    }
    scores["skip"] = 0.0
    if not warmup:
        action = max(scores, key=lambda key: (scores[key], key == "skip"))
        if scores[action] <= 0:
            action = "skip"
        return action, {"scores": scores, "probabilities": None}
    prior = {"skip": 0.30, "snippet": 0.60, "full_text": 0.05, "both": 0.05}
    prior.update(priors or {})
    if any(not math.isfinite(float(value)) or float(value) < 0 for value in prior.values()):
        raise ValueError("action priors must be finite and nonnegative")
    if prior.get("skip", 0) <= 0:
        raise ValueError("skip must retain a positive warm-up prior")
    if any(not math.isfinite(float(cost)) or float(cost) < 0 for cost in estimated_costs.values()):
        raise ValueError("estimated action costs must be finite and nonnegative")
    feasible = [key for key in scores if prior.get(key, 0) > 0]
    logits = {key: math.log(prior[key]) + scores[key] / temperature for key in feasible}
    maximum = max(logits.values())
    weights = {key: math.exp(value - maximum) for key, value in logits.items()}
    total = sum(weights.values())
    probabilities = {key: value / total for key, value in weights.items()}
    rng = rng or np.random.default_rng(42)
    action = rng.choice(list(probabilities), p=list(probabilities.values()))
    return str(action), {"scores": scores, "probabilities": probabilities}


def encode_context_texts(texts: list[str], *, model_name: str = "BAAI/bge-small-en-v1.5",
                         batch_size: int = 32, local_files_only: bool = False) -> np.ndarray:
    """Encode candidate title/snippet strings with frozen BGE; optional dependency."""
    global _EMBEDDER, _EMBEDDER_MODEL
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError("adaptive_linucb requires sentence-transformers; install it to run adaptive mode") from error
    if _EMBEDDER is None:
        options = {"local_files_only": True} if local_files_only else {}
        _EMBEDDER = SentenceTransformer(model_name, device="cpu", **options)
    elif _EMBEDDER_MODEL != model_name:
        raise ValueError("one EXPAND process cannot mix adaptive embedding models")
    _EMBEDDER_MODEL = model_name
    keys = [(model_name, " ".join(text.split())) for text in texts]
    missing = list(dict.fromkeys(key for key in keys if key not in _EMBEDDING_CACHE))
    if missing:
        encoded = _EMBEDDER.encode([key[1] for key in missing], batch_size=batch_size,
                                   normalize_embeddings=True, show_progress_bar=False)
        _EMBEDDING_CACHE.update({key: np.asarray(vector, dtype=float)
                                 for key, vector in zip(missing, encoded)})
    vectors = np.stack([_EMBEDDING_CACHE[key] for key in keys])
    if vectors.ndim != 2 or vectors.shape[1] != 384:
        raise ValueError(f"{model_name} must return 384-dimensional embeddings")
    return vectors


def similarity_features(candidate_embeddings: np.ndarray, positive_embeddings: np.ndarray,
                        negative_embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return clipped max cosine similarities for the two query groups."""
    candidates = np.asarray(candidate_embeddings, dtype=float)
    positive = np.asarray(positive_embeddings, dtype=float)
    negative = np.asarray(negative_embeddings, dtype=float)
    return (np.clip(candidates @ positive.T, 0, 1).max(axis=1),
            np.clip(candidates @ negative.T, 0, 1).max(axis=1))


def _organizations(text: str, model_name: str) -> set[str]:
    global _NER, _NER_MODEL
    cache_key = (model_name, text)
    if cache_key in _ORG_CACHE:
        return set(_ORG_CACHE[cache_key])
    try:
        import spacy
    except ImportError as error:
        raise RuntimeError("adaptive_linucb requires spaCy and its configured NER model") from error
    if _NER is None:
        try:
            _NER = spacy.load(model_name)
        except OSError as error:
            raise RuntimeError(f"adaptive NER model {model_name!r} is unavailable") from error
    elif _NER_MODEL != model_name:
        raise ValueError("one EXPAND process cannot mix adaptive NER models")
    _NER_MODEL = model_name
    generic = {"company", "supplier", "customer", "manufacturer", "group", "unknown",
               "the company", "our company", "n a", "na", "none"}
    found = {
        normalization_key(ent.text) for ent in _NER(text).ents
        if ent.label_ == "ORG" and len(ent.text.strip()) > 1
        and ent.text.strip().casefold() not in generic
    }
    _ORG_CACHE[cache_key] = found
    return set(found)


def build_document_contexts(results: list, entity_name: str, *, aliases: list[str],
                            known_companies: set[str], history: list[list[float]],
                            official_domains: list[str], config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Compute pre-action contexts and candidate vectors for one entity candidate pool."""
    selection = config["expansion"].get("selection") or {}
    texts = ["\n".join(part for part in (r.title, r.snippet) if part).strip() for r in results]
    if any(not text for text in texts):
        raise ValueError("adaptive candidate has no title or search snippet")
    positives = [
        f"Companies supplying products directly to {entity_name}",
        f"Lists of named suppliers and upstream procurement contracts for {entity_name}",
    ]
    negatives = [
        f"Customers purchasing products from {entity_name}",
        f"Aftermarket replacement parts compatible with {entity_name} products",
        f"Generic supplier requirements and procurement guidelines without named suppliers for {entity_name}",
    ]
    instruction = "Represent this sentence for searching relevant passages: "
    embedding_input = [*texts, *(instruction + q for q in positives + negatives)]
    embeddings = encode_context_texts(
        embedding_input, model_name=selection.get("embedding_model", "BAAI/bge-small-en-v1.5"),
        batch_size=int(selection.get("embedding_batch_size", 32)),
        **({"local_files_only": True} if config.get("_benchmark_offline") else {}),
    )
    n = len(texts)
    supplier, irrelevant = similarity_features(embeddings[:n], embeddings[n:n + 2], embeddings[n + 2:])
    prior = np.asarray(history, dtype=float)
    aliases_norm = [re.sub(r"\W+", " ", name.casefold()).strip() for name in [entity_name, *aliases] if name]
    known_norm = {normalization_key(name) for name in known_companies}
    official = {domain.casefold().lstrip(".") for domain in official_domains}
    contexts = []
    if not official:
        logger.warning("adaptive official-source map is empty; official-source features are zero")
    for i, result in enumerate(results):
        text_norm = re.sub(r"\W+", " ", texts[i].casefold()).strip()
        target_match = any(re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text_norm) for name in aliases_norm if name)
        host = urlsplit(result.canonical_source_url or result.url or "").hostname or ""
        is_official = any(host.casefold() == domain or host.casefold().endswith("." + domain)
                          for domain in official)
        if host and not is_official and host.casefold() not in _UNKNOWN_OFFICIAL_HOSTS:
            logger.info("adaptive official-source mapping is unknown for host %s", host)
            _UNKNOWN_OFFICIAL_HOSTS.add(host.casefold())
        rank = max(1, int(getattr(result, "provider_rank", 0) or i + 1))
        novelty = 1.0 if not len(prior) else float(1 - np.clip(embeddings[i] @ prior.T, 0, 1).max())
        orgs = _organizations(texts[i], selection.get("ner_model", "en_core_web_sm"))
        orgs = {name for name in orgs if name not in {"company", "supplier", "customer", "manufacturer"}}
        entity_novelty = (sum(name not in known_norm for name in orgs) / len(orgs)) if orgs else 0.0
        contexts.append(context_vector(
            supplier_relevance=float(supplier[i]), irrelevant_content=float(irrelevant[i]),
            target_match=target_match, official_source=is_official, search_rank=rank,
            document_novelty=novelty, entity_novelty=entity_novelty,
        ))
    return np.asarray(contexts), embeddings[:n]
