# SChain — Professor Colab EXPAND setup
#
# Seeds and expansion policy come from config/expansion.yaml.
# Do NOT set FOCAL_COMPANY / seeds = ["BMW"] in the normal path.
#
# Recommended:
#   CONFIG = "config/colab_professor_expand.yaml"   # includes expansion.yaml
#   MAX_DOCUMENTS = 150         # keep; 150 was not the bottleneck
#   FRESH_RUN = True
#   Runtime: expansion/colab overlay uses max_minutes=120
#   MAX_TIER: prefer expansion.yaml (5); override only if intentional
#
# v0.3 recall fixes active:
#   - broad "{entity} suppliers" query family
#   - zero-yield rescue up to per-tier max after primary target
#   - improved Google News title/publisher recovery
#   - funnel / fetch-failure / rejection telemetry in run_metrics.json

"""
# --- Colab cell sketch (not executed by the repo CI) ---

CONFIG = "config/colab_professor_expand.yaml"
MAX_DOCUMENTS = 150
FRESH_RUN = True
# Optional explicit overrides only:
# MAX_TIER = 5
# SEEDS_OVERRIDE = None

cmd = [
    "python", "-m", "schain", "expand",
    "--config", CONFIG,
    "--no-neo4j",
]
if FRESH_RUN:
    cmd.append("--restart")
if MAX_DOCUMENTS is not None:
    cmd.extend(["--max-documents", str(MAX_DOCUMENTS)])
"""
