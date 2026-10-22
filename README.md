# AiLabwork
# GitHub Repo Quality Scorer

A Streamlit app that scores the quality of your GitHub repositories the way a recruiter or admissions reviewer skims them: README, tests, project structure and commit history. Paste a profile or a single repo and get a score, a list of what to fix first, and an optional Gemini review of your README.

## Why the score is rule-based

The numeric score is **100% deterministic**. The same repo always gets the same score, so you can fix things and measure whether your changes helped. Gemini is used only to explain and suggest (README clarity, action plan, which repos to pin); it never changes the score.

## What it checks

| Category | Points | Checks |
|---|---|---|
| README | 30 | exists, length, headings, install/usage instructions, code examples, screenshot/diagram/demo |
| Tests & CI | 20 | test files present, CI workflow (skipped for notebook-heavy or very small repos) |
| Project structure | 20 | `.gitignore`, dependency file, license, code organised in folders, no junk/secret files committed |
| Commit hygiene | 20 | commit count, descriptive messages, history spread over time, recent activity |
| Repo metadata | 10 | description, topics |

When a category does not apply (for example, tests for a notebook-only repo) it is dropped and the remaining points are rescaled to 100.

## Setup

```bash
git clone https://github.com/<your-username>/<this-repo>.git
cd <this-repo>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Keys (all optional)

- **Gemini API key** enables the AI README review. Get a free key at [aistudio.google.com](https://aistudio.google.com).
- **GitHub token** raises the API limit from about 60 to 5,000 requests/hour. A token with no scopes is enough for public repos.

Provide them in the sidebar, or as environment variables:

```bash
export GEMINI_API_KEY="your-key"
export GITHUB_TOKEN="your-token"
export GEMINI_MODEL="gemini-3.5-flash"   # optional override
```

You can also use `.streamlit/secrets.toml`. Never commit it.

## Usage

```bash
streamlit run app.py
```

Then paste any of these:

- `https://github.com/username` (profile: analyses your most recently pushed repos)
- `https://github.com/username/repo` (a single repository)
- `username` or `username/repo`

## Limitations

- **Heuristics, not truth.** A repo can pass every check and still be bad, or fail some and be excellent. The score has not been validated against real recruiter decisions.
- **Commit messages** are judged only on the 100 most recent commits, by simple rules (too short, "update", "Update README.md", and similar).
- **Secret detection is filename-only.** It flags files like `.env` or `*.pem`; it does not read file contents.
- **API limits.** Each repo costs about 3 GitHub API requests. Results are cached for 15 minutes.
- **Notebook repos** skip the test/CI checks, and tiny scripts are not expected to have an elaborate structure.
- The AI review sees only a README excerpt, so it cannot judge the code itself.

## Roadmap

- [ ] Validate the score: hand-label 20 repos (strong / medium / weak) and measure rank correlation with the score
- [ ] Add a screenshot or demo GIF to this README
- [ ] Unit tests for the scoring functions
- [ ] Optional code-level signals (function length, docstrings, linting)
- [ ] Before/after comparison to track improvement over time

## Tech stack

Python, Streamlit, GitHub REST API, Google Gemini API (`google-genai`).
