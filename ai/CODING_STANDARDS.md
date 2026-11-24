# Scouter Project Coding Standards and Agent Guidelines

## 1. Code Generation and Chat Delivery Standards
* **Clean Copy-Paste Guarantee & No Fenced Code Blocks:** Whenever code, terminal commands, or shell scripts are presented in conversation, NEVER use fenced code blocks (triple backticks ```), as the CLI/UI renderer automatically injects line numbers (1, 2, 3) into fenced blocks. Always deliver code as raw, un-fenced plain text.
* **Proper Comment Syntax:** Comments are permitted and encouraged, but they must strictly use valid syntax for the target interpreter (e.g. `#` for Python/Bash). To prevent Markdown renderers from misinterpreting comment lines as HTML headings (which strips the `#` symbol), format comments with `#--` or ensure the `#` prefix is preserved intact so the copied code is 100% syntactically valid.
* **Language Consistency:**
  - All source code, docstrings, variable names, directory paths, file names, commit messages, and repository markdown files must be strictly in **English**.
  - User-facing conversational responses are conducted in **Italian**.
* **Plain Text Math:** Never use LaTeX formatting (`$...$`, `\frac`, etc.) in CLI output or documentation. Use standard ASCII notations (`>=`, `<=`, `!=`, `->`, `x`, `Delta`).

## 2. Execution and Environment Workflow
* **Safe Enterprise Execution:** The local repository runs on an enterprise system. Never initiate background SSH tunnels, proxy executables, or third-party tunneling binaries on the local host.
* **Compute Offloading:** Compute-intensive operations (video downloading, deep learning inference, YOLO tracking, Re-ID embedding extraction, and video slicing) run in cloud environments (Google Colab / Cloud GPU).
* **Automated Runner (`run_pipeline.py`):** Development prioritizes single-entrypoint autonomous execution scripts over fragmented multi-cell notebook sequences to minimize manual bridging between agents.

## 3. Git Workflow and Commit Strategy
* **Local-First Commits:** Structure modifications into clear, surgical local commits as logical workunits or milestones are accomplished.
* **Batched Pushing Mandate:** NEVER execute `git push` on every minor edit or micro-fix. Accumulate local commits and push to remote origin only in consolidated batches when a full feature milestone is completed or when explicitly instructed by the user.
