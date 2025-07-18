Of course. Here is your README, formatted into clean and readable Markdown without altering any of the original text.

---

# Fine-tuning Qwen1.5-4B with Advanced Preference Optimization Methods

This report details the process of fine-tuning the `Qwen1.5-4B` model using three distinct techniques: `QLORA`, `GRPO`, and `ORPO`. The experiments were conducted using Kaggle and Google Colab notebooks. The goal was to enhance the model's mathematical reasoning capabilities.

**Models available on Hugging Face:**

- **QLORA Model:** [Shrayon/mathmind_qlora](https://huggingface.co/Shrayon/mathmind_qlora)
- **GRPO Model:** [Shrayon/mathmind_grpo](https://huggingface.co/Shrayon/mathmind_grpo)
- **ORPO Model:** [Shrayon/mathmind_orpo](https://huggingface.co/Shrayon/mathmind_orpo)

## 1. QLORA (Quantized Low-Rank Adaptation) Fine-tuning

### 1.1. What is QLORA?

QLORA (Quantized Low-Rank Adaptation) is a highly efficient fine-tuning technique that makes it possible to tune large language models on consumer-grade hardware. It works by quantizing the main model to a lower precision (like 4-bit) to reduce memory usage, freezing its weights, and then training a small number of "adapter" matrices, a method known as Low-Rank Adaptation (LoRA)., Gradients are backpropagated through the frozen, quantized model into these LoRA adapters. This approach significantly lowers the GPU memory requirements and speeds up training while preserving the performance levels of full 16-bit fine-tuning.,

### 1.2. Dataset and Preprocessing

- **Composition:** The dataset was a strategic mix of reasoning and non-reasoning data to build a versatile model.
  - **Reasoning Data:** `unsloth/OpenMathReasoning-mini`
  - **Chat/Non-reasoning Data:** `mlabonne/FineTome-100k`
- **Merging Strategy:** The datasets were merged with a 75% split to favor reasoning. We sampled 25% of the reasoning dataset and 75% of the chat-based dataset. This was done to ensure the model retained strong foundational reasoning capabilities while improving its conversational abilities.
- **Total Size:**
  - Reasoning samples: 19,252
  - Non-reasoning samples: 6,417
  - **Total:** 25,669 examples
- **Formatting:** All data was formatted into the ChatML style. Proper end tokenization and padding were strictly enforced to prevent the model from generating infinitely looping responses.

### 1.3. Training Configuration

The training process was meticulously tracked using Weights & Biases (`wandb`). The following parameters were chosen to balance performance and resource constraints.

| Parameter                   | Value                  | Justification                                                                                                                                                                                                                                                                                                                       |
| :-------------------------- | :--------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Train/Validate Split**    | 80/20                  | Standard practice to monitor for overfitting on unseen data during training.                                                                                                                                                                                                                                                        |
| **Epochs**                  | 3                      | Determined as the "sweet spot" after experimenting with 1, 2, and 5 epochs to maximize learning without overfitting.                                                                                                                                                                                                                |
| **LoRA Rank**               | 32                     | A balance between model expressiveness and the number of trainable parameters. A higher rank allows for more complex adaptations.                                                                                                                                                                                                   |
| **Max Sequence Length**     | 2048                   | Chosen to accommodate the long reasoning traces and complex problem statements often found in mathematical questions.                                                                                                                                                                                                               |
| **Batch Size**              | 2                      | The maximum size that could fit on the available GPU.                                                                                                                                                                                                                                                                               |
| **Gradient Accumulation**   | 4                      | This results in an effective batch size of 8 (2 x 4), allowing for more stable gradients without increasing memory usage.                                                                                                                                                                                                           |
| **Gradient Checkpointing**  | Enabled                | A trade-off where computation is slightly increased to significantly reduce VRAM usage, enabling training with a larger sequence length.                                                                                                                                                                                            |
| **Logging / Saving / Eval** | 100 / 1200 / 600 steps | Frequent logging provides insight, while periodic saving and evaluation ensure progress is not lost and the best model is identified. With ~20.5k training examples and an effective batch size of 8, each epoch is ~2567 steps, making the total training ~7700 steps. This schedule is well-calibrated for the training duration. |
| **Warmup Ratio**            | 0.01                   | A small warmup phase where the learning rate gradually increases helps stabilize training at the beginning.                                                                                                                                                                                                                         |
| **LR Scheduler**            | Cosine                 | The learning rate is smoothly decreased over the training run, which often leads to better final model performance.                                                                                                                                                                                                                 |
| **Optimizer**               | Paged Adam 8-bit       | A memory-efficient optimizer specifically designed for QLORA, which "pages" optimizer states to the CPU to save GPU memory.                                                                                                                                                                                                         |
| **Early Stopping**          | Patience = 5           | Training halts if validation loss does not improve for 5 consecutive evaluations, preventing overfitting and saving the best-performing model checkpoint.                                                                                                                                                                           |

### 1.4. Model Artifacts

The final trained LoRA adapters, the merged model, and a GGUF version (for CPU-based inference with `llama.cpp`) were successfully pushed to the `Shrayon/mathmind_qlora` repository on Hugging Face.

## 2. GRPO (Group Relative Policy Optimization) Fine-tuning

### 2.1. What is GRPO?

Group Relative Policy Optimization (GRPO) is an advanced reinforcement learning technique that enhances preference optimization., Unlike methods that compare a single "chosen" response against a single "rejected" one (like DPO), GRPO evaluates a group of generated responses for a given prompt simultaneously. It then uses relative rewards within this group to guide the model, creating a more stable and robust learning signal that leads to better overall performance.,

### 2.2. Dataset and Reward Engineering

- **Dataset:** To manage GPU constraints, we started with a smaller subset (1k examples) of the `unsloth/OpenMathReasoning-mini` dataset. Once the methodology proved solid, the dataset was expanded with more complex reasoning examples from `open-r1/DAPO-Math-17k-Processed`. The data was formatted in ChatML style.
- **Initial SFT:** A quick Supervised Fine-tuning (SFT) run was performed first (rank 16, 1 epoch, learning rate 2e-4) to align the model with the dataset format and establish a working baseline.
- **Custom Reward Functions:** The core of GRPO lies in its reward mechanism. We defined several functions to score generated responses:
  - `match_format_exactly`: +3 points if the output perfectly matches the desired format (e.g., includes reasoning traces).
  - `match_format_approximately`: Graded rewards if certain key segments (like reasoning start/end tags) are present.
  - `check_final_answer`: A detailed reward scheme for mathematical correctness:
    - +5 points for a perfectly correct final answer.
    - +3.5 points if the answer is correct but has minor formatting issues (e.g., extra spaces).
    - +2 to +1 points if the answer is numerically close to the correct one.
    - -2.5 to -4.5 points for incorrect answers, with larger penalties for greater deviation.

### 2.3. Training Configuration

The training was configured to leverage the `vLLM` engine for fast generation and `trl`'s `GRPOConfig` for the training loop.

| Parameter                     | Value                            | Justification                                                                                                                                                                      |
| :---------------------------- | :------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **vllm_sampling_params**      | `min_p=0.1, top_p=1.0, top_k=-1` | These settings encourage diverse generations by using nucleus sampling (`min_p`), which is essential for creating a varied group of candidate responses for preference comparison. |
| **temperature**               | 1.0                              | A high temperature further increases the randomness and diversity of the generated responses.                                                                                      |
| **learning_rate**             | 5e-6                             | A low learning rate is crucial for stable fine-tuning in preference optimization to avoid catastrophic forgetting of the model's pre-trained knowledge.                            |
| **lr_scheduler_type**         | linear                           | A simple and effective scheduler that decreases the learning rate linearly from its initial value to zero.                                                                         |
| **optim**                     | `adamw_8bit`                     | A memory-efficient AdamW optimizer suitable for resource-constrained environments.                                                                                                 |
| **Batch Size / Grad. Accum.** | 1 / 4                            | An effective batch size of 4 provides smoother gradient updates, which is important for the stability of RL-based methods.                                                         |
| **num_generations**           | 4                                | For each prompt, generate 4 different responses. This group of responses is then used to calculate relative preference scores.                                                     |
| **Max Steps**                 | 100                              | The training was set for a limited number of steps to demonstrate the method's effectiveness within available compute.                                                             |

### 2.4. Model Artifacts

The resulting model, which learned from group-wise preference feedback, was pushed to the `Shrayon/mathmind_grpo` repository, along with its GGUF version.

## 3. ORPO (Odds Ratio Preference Optimization) Fine-tuning

### 3.1. What is ORPO?

Odds Ratio Preference Optimization (ORPO) is a novel and efficient technique that merges supervised fine-tuning (SFT) and preference alignment into a single, monolithic process.,, Unlike DPO or RLHF which are separate alignment steps, ORPO modifies the standard training objective. It adds a penalty term based on the odds ratio, which weakly penalizes rejected responses while strongly rewarding chosen ones., This allows the model to learn the desired task and align with human preferences simultaneously, making it more efficient than multi-stage pipelines.,

### 3.2. Dataset and Prompting

- **Dataset:** The primary dataset used was `blesspearl/orpo-optimized-math-qa`. This dataset is structured for preference tuning, containing a prompt, a high-quality "accepted" response generated by GPT-4o, and a lower-quality "rejected" response from Mistral Small.
- **Ongoing Work:** A custom dataset is under construction using the well-known `GSM8K` benchmark. This involves generating "accepted" answers from a powerful teacher model (like GPT-4) and "rejected" answers from the base `Qwen1.5-4B` model itself. This process is currently paused due to a lack of a GPT API key.
- **Formatting:** The data was formatted using the Alpaca instruction-following style, which is a common standard for this type of fine-tuning.

### 3.3. Training Configuration

The training was conducted using the `ORPOTrainer` from the TRL library, patched by Unsloth for maximum efficiency.

| Parameter                     | Value                                                                     | Justification                                                                                                                                                                                                                                                                                    |
| :---------------------------- | :------------------------------------------------------------------------ | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Batch Size / Grad. Accum.** | 2 / 4                                                                     | An effective batch size of 8 was used for stable training.                                                                                                                                                                                                                                       |
| **beta**                      | 0.1                                                                       | This is the key hyperparameter in ORPO. It controls the trade-off between the standard SFT loss and the preference alignment loss. A value of 0.1 gives more weight to learning the task (SFT) while still applying a gentle penalty to rejected answers, which is a recommended starting point. |
| **optim**                     | `adamw_8bit`                                                              | A memory-efficient optimizer.                                                                                                                                                                                                                                                                    |
| **lr_scheduler_type**         | linear                                                                    | A simple and reliable learning rate schedule.                                                                                                                                                                                                                                                    |
| **Epochs**                    | 1                                                                         | A single epoch is often sufficient for preference tuning, as the model primarily learns to distinguish between good and bad responses rather than learning new knowledge from scratch.                                                                                                           |
| **Max Lengths**               | `max_length=2048`, `max_prompt_length=1024`, `max_completion_length=1024` | The sequence lengths were divided to ensure both the prompt and the completion could fit within the model's context window.                                                                                                                                                                      |

### 3.4. Model Artifacts

The final ORPO-tuned model, tokenizer, and GGUF version were saved and pushed to the `Shrayon/mathmind_orpo` repository.

## 4. Evaluation and Results

The models were evaluated on a sampled subset of 100 examples from the `GSM8K` dataset. Performance was measured using the BLEU score (to assess textual similarity to the reference solution) and `pass@1` (to measure functional correctness of the final answer).

| Method  | GPU Used             | Training Time | BLEU | pass@1 |
| :------ | :------------------- | :------------ | :--- | :----- |
| `QLoRA` | NVIDIA T4 (Colab)    | ~7 hours      | 39.2 | 41 %   |
| `GRPO`  | NVIDIA P100 (Kaggle) | ~10 hours     | 45.3 | 53 %   |
| `ORPO`  | NVIDIA T4 (Colab)    | ~6 hours      | 43.7 | 49 %   |

> **Note:** Training times are estimates for the specified datasets and configurations. GRPO is slower due to the on-the-fly generation of multiple responses per prompt. Training was continued till overfitting beyond the patience limit occurred. Detailed graph based results from wandb are in the results folder.

## 5. Conclusion

Based on the tested parameters, the model fine-tuned with `GRPO` performed the best.

The `GRPO` method achieved the highest BLEU score and, more importantly, the highest `pass@1` rate. This suggests that its more complex, group-based reward mechanism, which provided nuanced feedback on both formatting and correctness, was more effective at teaching the model to produce accurate and well-structured mathematical reasoning compared to the direct supervised approach of `QLORA` and the simpler pairwise preference of `ORPO`. While more computationally intensive, the investment yielded superior results in functional correctness for this task.
