# Project Proposal: On-Device Personalized Agent using Fine-tuned Qwen3-0.6B

Brandon Howell

## 1. Introduction

**Scope:** This project aims to create a personalized AI assistant capable of running directly on an Android smartphone. The core involves fine-tuning a small, efficient Large Language Model (LLM), specifically Qwen3-0.6B[^qwen], with personal user data. Subsequently, this personalized model will be integrated into an experimental Android agent framework (potentially a fork of Gosling[^gosling]) to enable it to perform tasks by interacting with other applications on the device.

[^qwen]: https://huggingface.co/Qwen/Qwen3-0.6B
[^gosling]: https://github.com/block/gosling

**Problem:** Current large language models, while powerful, are typically generic and operate in the cloud, raising privacy concerns and lacking deep personalization. Running large models locally on resource-constrained devices like smartphones is computationally challenging. Furthermore, enabling these models to interact meaningfully with the device's applications and data in an automated fashion (i.e., act as agents) presents significant technical hurdles in the mobile environment. This project addresses the need for a private, personalized, and capable AI assistant that lives entirely on the user's device.

**Interest & Non-Triviality:** This project is interesting because it tackles the intersection of several cutting-edge AI domains that I've been following the news of closely for a while now:
*   **On-Device AI:** Deploying capable LLMs on mobile phones pushes the boundaries of model optimization and efficient inference.
*   **LLM Personalization:** Fine-tuning with personal data explores methods to make AI truly tailored to an individual user's context, habits, and preferences.
*   **Agentic AI on Mobile:** Integrating an LLM with device automation frameworks like Gosling explores the practical challenges and potential of creating autonomous agents in the complex and varied Android ecosystem.
*   **Parameter-Efficient Fine-Tuning (PEFT):** Experimenting with techniques like adapters offers insights into efficient model adaptation for resource-constrained environments.

The combination of fine-tuning a very recent small LLM, deploying it locally, integrating personal context, and building agentic capabilities makes this project ambitious and non-trivial, offering ample room for experimentation and learning.

`[Placeholder for Figure 1: Conceptual diagram showing a phone running a personalized LLM that interacts with apps like Calendar, Messages, etc.]`

## 2. Related Work

The development of personalized, on-device AI agents draws upon research in several areas:

*   **Small Language Models (SLMs):** The trend towards smaller, yet capable, language models is crucial for on-device deployment. Models like Phi-3 (Microsoft), Gemma (Google), and the chosen Qwen3-0.6B (Alibaba) represent efforts to achieve high performance with significantly fewer parameters than models like GPT-3/4. Their smaller size makes them candidates for mobile inference. The Qwen3 series specifically notes optimization for resource-constrained environments ([Qwen Team, 2024](https://qwenlm.github.io/blog/qwen3/)).
*   **On-Device LLM Inference:** Techniques for efficiently running LLMs on mobile devices are critical. This includes model quantization (reducing the precision of model weights), optimized inference engines (like `llama.cpp`, MediaPipe LLM Inference, ONNX Runtime), and leveraging mobile NPUs/GPUs. Research often focuses on minimizing latency and memory footprint while preserving model quality (e.g., [Kim et al., 2024, "MobileLLM"](https://arxiv.org/abs/2307.06506) - *Note: Find a real relevant paper if possible, this one is just an example structure*).
*   **LLM Fine-Tuning for Personalization:** Adapting pre-trained LLMs to specific domains or user data is a common practice. While full fine-tuning modifies all parameters, **Parameter-Efficient Fine-Tuning (PEFT)** methods have gained prominence. Techniques like Adapter Tuning ([Houlsby et al., 2019](https://arxiv.org/abs/1902.00751)), LoRA ([Hu et al., 2021](https://arxiv.org/abs/2106.09685)), and QLoRA ([Dettmers et al., 2023](https://arxiv.org/abs/2305.14314)) allow adaptation by training only a small number of additional parameters. This is particularly relevant for on-device scenarios, reducing computational cost and storage requirements for personalized models ([Raschka, 2024](https://magazine.sebastianraschka.com/p/finetuning-llms-with-adapters)).
*   **LLM-Powered Agents:** Frameworks enabling LLMs to use tools, plan, and execute tasks have emerged (e.g., LangChain, AutoGPT). Applying these concepts to mobile environments involves interacting with apps and system services. Projects like Gosling ([Block, Inc.](https://github.com/block/gosling)) specifically explore agentic capabilities on Android, using platform features like Accessibility Services or custom inter-app communication protocols (like their proposed "mobile MCP") to automate tasks. This relates to broader research on autonomous agents and tool augmentation for LLMs (e.g., [Schick et al., 2023, "Toolformer"](https://arxiv.org/abs/2302.04761)).

## 3. Method

This project will be developed in two main phases:

**Phase 1: Personalized On-Device LLM Fine-tuning**

1.  **Model Selection:** The core model will be Qwen3-0.6B, chosen for its small size and state-of-the-art performance within its parameter class. ([Qwen3-0.6B Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B)).
2.  **Data Collection & Preparation:** A personal dataset will be curated. This may include (but is not limited to):
    *   Calendar entries (structure, common events, timings).
    *   Anonymized or synthesized communication logs (SMS/email snippets reflecting personal style, common contacts).
    *   Personal notes or documents (FAQs about oneself, preferences, routines).
    *   App usage patterns (e.g., frequently used apps for specific tasks).
    *   Location habits (e.g., common places like "home", "work", "gym").
    *   *Crucially, this data will need careful anonymization/synthesization and curation to be useful and privacy-preserving.* The format will likely be structured prompts and desired responses (instruction fine-tuning format).
3.  **Fine-tuning Implementation:** Two fine-tuning approaches will be explored using libraries like Hugging Face `transformers` and `peft`:
    *   **Full Fine-tuning:** Update all weights of the Qwen3-0.6B model on the curated personal dataset.
    *   **Adapter-based Fine-tuning (PEFT):** Freeze the base Qwen3-0.6B model and train lightweight adapter modules (e.g., LoRA or the method from [Houlsby et al., 2019](https://arxiv.org/abs/1902.00751)).
4.  **On-Device Deployment:** The fine-tuned models (both versions) will be prepared for on-device execution. This will likely involve:
    *   Quantization (e.g., to 4-bit using GGUF format).
    *   Using an on-device inference engine compatible with Qwen models and Android, such as `llama.cpp`'s Android bindings or potentially adapting other frameworks if necessary.
    *   Building a simple Android application wrapper to load the model and allow text-based interaction for initial testing.

`[Placeholder for Figure 2: Diagram illustrating the fine-tuning process (Data -> Full Fine-tuning vs PEFT -> Quantization -> On-Device Model)]`

**Phase 2: Agentic Integration with Gosling**

1.  **Framework Setup:** Fork the Gosling Android agent repository ([Gosling GitHub](https://github.com/block/gosling)). Familiarize with its architecture, particularly how it invokes LLMs and interacts with device capabilities (Accessibility Services, Intents, potential "mobile MCP").
2.  **Model Integration:** Modify the forked Gosling code to use the personalized, locally deployed Qwen3-0.6B model (likely the PEFT version due to efficiency) as its reasoning engine. This will involve adapting the LLM API calls within Gosling to interface with the chosen on-device inference engine.
3.  **Tool Definition & Use:** Explore Gosling's mechanisms for defining and using tools. This might involve:
    *   Leveraging existing Gosling capabilities triggered via natural language prompts.
    *   Potentially implementing a simple custom "mobile MCP" provider app (as shown in the Gosling README example) to expose a new capability (e.g., retrieving a piece of personal info directly).
4.  **Task Automation:** Define and test simple, multi-step tasks that require the agent to use the personalized LLM and interact with other apps via Gosling's capabilities (e.g., "Check my schedule for tomorrow afternoon", "Draft a text to Mom in my usual style saying I'll be late").

`[Placeholder for Figure 3: Diagram showing Gosling architecture with the custom fine-tuned LLM integrated, interacting with Android APIs/Apps.]`

## 4. Evaluation

Success will be evaluated based on experimentation depth and insights gained, rather than solely on flawless execution.

**Phase 1 Evaluation (Personalized LLM):**

*   **Dataset:** A held-out set of prompts based on the personal data domain (e.g., questions about schedule, preferences, communication style).
*   **Metrics:**
    *   **Qualitative Assessment:** Subjective evaluation of the model's responses for personalization (does it know my context?), accuracy (correct information), coherence, and tone compared to the base Qwen3 model.
    *   **Quantitative Assessment (Exploratory):**
        *   Perplexity on the held-out personal dataset.
        *   Resource Usage: Measure model storage size (base vs. full fine-tune vs. base + adapter), inference latency, and peak memory usage on the Android device for both fine-tuning approaches.
    *   **Comparison:** Directly compare the effectiveness and resource trade-offs between the fully fine-tuned model and the adapter-based model.

**Phase 2 Evaluation (Agentic Capabilities):**

*   **Dataset:** A set of defined, multi-step tasks requiring interaction with device features or apps (e.g., "Check weather and message John about going to the park", "Summarize my unread emails from Alice", "Set a reminder for my usual evening routine").
*   **Metrics:**
    *   **Task Completion Rate:** Percentage of tasks successfully completed by the agent.
    *   **Qualitative Assessment:** Evaluate the agent's planning ability, tool usage effectiveness, error handling, and overall usefulness for the defined tasks. Does the personalization from Phase 1 demonstrably improve task execution (e.g., using correct contact names, understanding implicit context)?
    *   **Efficiency (Observational):** Note the time taken and number of steps/interactions required for task completion.

## 5. Milestones

*(Left blank as requested - you will fill this in with your specific timeline, e.g., Week 3: Data curation complete, Week 5: Initial fine-tuning experiments running, Week 7: Model deployed on device, Week 9: Gosling integration started, Week 11: Agent task evaluation)*

## 6. Task Assignments

I'm working solo for this project since there was some miscommunication with groups, so don't have to worry about splitting up tasks.

