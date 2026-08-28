# Supply Chain KG V1 — Claude Starter Pack

This package contains the current implementation brief and development rules for the Supply Chain Knowledge Graph V1 project.

## Start here

Claude should read in this order:

1. `CLAUDE.md` — mandatory development workflow and engineering rules.
2. `docs/SupplyChain_KG_V1_Architecture_Brief_Final.pdf` — compact project architecture.
3. `PROJECT_DECISIONS.md` — frozen V1 decisions and scope.
4. `PROJECT_STRUCTURE.md` — proposed simple repository structure.
5. `config/config.example.yaml` — example configuration shape.

The project is intentionally designed to be simple, traceable, and based on official/common packages instead of custom reimplementation.

Work autonomously within the frozen V1 decisions in `CLAUDE.md`. Keep each change simple, traceable, based on common/official packages where possible, and avoid unrelated or cosmetic additions.

## Repository location

This starter pack is intended to be extracted **directly into the root of the private `SChain` repository**:

`git@github.com:ErfanBabakhani/SChain.git`

After extraction, `CLAUDE.md`, `README.md`, `config/`, `docs/`, and future `src/` should all be directly under the `SChain` root. Do not keep this starter pack inside an extra nested folder.

For Git setup and private-repository authentication, see `GIT_SETUP.md`.
