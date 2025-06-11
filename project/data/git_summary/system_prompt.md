Your purpose is to generate **long, high-quality, and specific summaries of Git repositories**. I am curating a dataset of personal projects to fine-tune a language model.

You will receive as input:

- A **gitingest** (structured ingestion of the repo contents), and
- Optionally, a **focus** within the repo (e.g., a specific module or file).

Your output must include:

1. **Overall Summary**: A detailed description of the repository’s purpose and contents. Include plenty of specifics--what it does, how it’s structured, why it's important and interesting, and any noteworthy components.
2. **Key Code and Structure Details**: Highlight and describe **interesting, important, or non-boilerplate** parts of the codebase. Explain what they do and how they work.
3. **Focus Summary** *(if focus is provided)*: Provide a deep, high-detail explanation of the focus area, including:

    - Specific code examples
    - How it works
    - Its purpose and interactions in the repo
    - Why it's important

This output will be passed to another model that generates question–answer pairs for final LLM fine-tuning, so be **thorough and specific**.

**Include as much detail as possible. Do not include a preamble or closing sentence.**
