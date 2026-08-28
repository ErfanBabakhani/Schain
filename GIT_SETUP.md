# Git Setup for SChain

Repository:

- SSH: `git@github.com:ErfanBabakhani/SChain.git`
- HTTPS: `https://github.com/ErfanBabakhani/SChain.git`
- Visibility: private

## Recommended: use SSH

SSH is the simplest option for a private repository if your machine already has a GitHub SSH key.

### If the repository is not cloned yet

```bash
git clone git@github.com:ErfanBabakhani/SChain.git
cd SChain
```

Extract the starter ZIP **inside this directory** so that `CLAUDE.md`, `README.md`, `config/`, and `docs/` are directly under `SChain/`.

Then:

```bash
git status
git add .
git commit -m "Initialize supply-chain KG V1"
git push -u origin main
```

If the default branch is not `main`, use the branch already configured in the repository.

### If `SChain` already exists locally

From the repository root:

```bash
cd SChain
git remote -v
git status
```

If `origin` is missing:

```bash
git remote add origin git@github.com:ErfanBabakhani/SChain.git
```

If `origin` points somewhere else:

```bash
git remote set-url origin git@github.com:ErfanBabakhani/SChain.git
```

Then commit and push normally:

```bash
git add .
git commit -m "Initialize supply-chain KG V1"
git push
```

## Private repository authentication

For SSH access, the local machine/user running Git needs an SSH key that is authorized on the GitHub account with access to this private repository.

Quick check:

```bash
ssh -T git@github.com
```

If no SSH key exists, create one:

```bash
ssh-keygen -t ed25519 -C "your-email@example.com"
```

Then add the public key (normally `~/.ssh/id_ed25519.pub`) to GitHub under:

`Settings -> SSH and GPG keys -> New SSH key`

Do **not** copy the private key into the repository.

## Git identity

If commits fail because Git does not know who you are:

```bash
git config --global user.name "Your Name"
git config --global user.email "your-github-email@example.com"
```

## If Claude is only writing local code

Claude does not need GitHub credentials just to edit files in the local `SChain` directory.

If Claude itself is expected to run `git push`, the environment where Claude runs must also have access to the SSH key / SSH agent for this private repository.

## HTTPS alternative

If SSH cannot be used, HTTPS requires authenticated GitHub access, typically through GitHub CLI or a personal access token with permission to the private repository. Do not store the token in the repository or `.env`.

## Before the first push

Check:

```bash
git status
git remote -v
git diff --cached
```

Make sure no `.env`, credentials, downloaded datasets, checkpoints, local databases, model weights, or caches are staged.
