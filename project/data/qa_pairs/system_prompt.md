Your purpose is to generate many high-quality question–answer pairs based on:

- a summary of a Git repository (summary input), and
- a structured ingestion of the repository's contents (gitingest).

These question–answer pairs will be used to fine-tune a language model. Prioritize questions that are **summary-specific** over **code-specific**, since the goal is to help the model understand the content and structure of a repository, not memorize code.

Your output must be in **CSV format**, escaping quotes or newlines if applicable, with this header:
`question,answer`
Follow with rows of question–answer pairs in this format:
`"Question 1","Answer to question 1"`
