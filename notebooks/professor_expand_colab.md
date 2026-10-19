# SChain professor EXPAND notebook

The earlier file was only a non-executable cell sketch and defaulted every session
to a fresh `--no-neo4j` run. The runnable replacement is:

- [`SChain_Expand_Persistent_Save.ipynb`](SChain_Expand_Persistent_Save.ipynb)

It uses `config/colab_professor_expand.yaml` as the policy overlay, enables
incremental Neo4j Save, keeps live SQLite on `/content`, and snapshots the same
experiment to Google Drive for later resume. See
[`docs/SAVE_AND_COLAB_RESUME.md`](../docs/SAVE_AND_COLAB_RESUME.md) for setup and
recovery semantics.
