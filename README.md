# Gordon Ramsay Persona Fine-Tuning via DPO

## 📖 Overview
This project demonstrates how to fine-tune a Large Language Model to adopt a highly specific persona—in this case, the rude, aggressive, and critical persona of chef Gordon Ramsay. The project leverages **Direct Preference Optimization (DPO)** to align the model by teaching it to prefer "Ramsay-style" answers over standard "polite" answers. 

The pipeline uses `unsloth` for memory-efficient training of the `Mistral-7B-Instruct-v0.3` model and includes a robust evaluation phase using an LLM-as-a-Judge and vector embeddings.

---

## ⚙️ Tech Stack & Tools

| Component | Technology |
| :--- | :--- |
| **Base Model** | `unsloth/mistral-7b-instruct-v0.3-bnb-4bit` |
| **Training Frameworks** | Unsloth, Hugging Face `trl` (DPOTrainer), `peft` (LoRA) |
| **Data Handling** | Pandas, Hugging Face `datasets` |
| **LLM Evaluator** | `nvidia/nemotron-3-nano-30b-a3b` (via NVIDIA API) |
| **Embedding Evaluator** | SentenceTransformers (`all-MiniLM-L6-v2`) |
| **Visualization** | Matplotlib, Seaborn |

---

## 🛠️ Project Pipeline

### 1. Data Preprocessing
*   **Dataset Structuring:** The dataset consists of 500 training and 100 testing Q&A pairs. Each row contains a `Question`, a `Polite` answer, and a `Ramsay` answer.
*   **DPO Formatting:** The data is transformed into a preference dataset featuring a system prompt, where the `Ramsay` response is marked as `chosen` and the `Polite` response is marked as `rejected`.

### 2. Model Training (DPO)
*   **LoRA Configuration:** Low-Rank Adaptation (LoRA) is applied to linear layers (`q_proj`, `k_proj`, `v_proj`, etc.) with a rank (`r`) of 64 to ensure efficient training.
*   **DPO Trainer:** The model is fine-tuned using the `DPOTrainer` to maximize the reward margin between the chosen and rejected responses. 

### 3. Inference & Comparison
The script generates and compares answers under three different conditions to validate the fine-tuning:
1.  **Pure LLM (No Prompt):** The base model answering without any persona instructions.
2.  **Pure LLM (With Prompt):** The base model answering with a system prompt instructing it to act like Gordon Ramsay.
3.  **DPO Fine-Tuned Model:** The newly trained model answering with the system prompt.

### 4. Evaluation 
Evaluating generative persona alignment is subjective, so the project uses two automated methods:
*   **LLM-as-a-Judge:** Uses an NVIDIA Nemotron model with a strict prompt to score the DPO model's output against the "Gold Standard" Ramsay reference. It scores from 1 to 10 on two metrics:
    *   *Semantic Similarity:* Does it convey the same sentiment as the reference?
    *   *Factuality:* Is the technical explanation consistent within the persona?
*   **Cosine Similarity:** Uses `SentenceTransformer` to create vector embeddings of the generated text and the reference text, calculating the cosine similarity score.
*   **Visualizations:** Includes Seaborn count plots charting the distribution of the Semantic Similarity and Factuality scores.

---

## 🚀 How to Use the Code
1. Ensure you are running this in an environment with a GPU (e.g., Google Colab T4).
2. Install the required dependencies (specifically `unsloth`, `trl`, and `sentence_transformers`).
3. Place your raw `.csv` datasets in the root directory.
4. Run the notebook sequentially to format the data, train the model, and evaluate the outputs. The fine-tuned model will be saved as `final_model_mistral.zip`.
