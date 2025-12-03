---
title: Llama Ui
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.42.0
app_file: chat_ui/app.py
pinned: false
hf_oauth: true
hf_oauth_scopes:
- inference-api
license: apache-2.0
---

An example chatbot using [Gradio](https://gradio.app), [`huggingface_hub`](https://huggingface.co/docs/huggingface_hub/v0.22.2/en/index), and the [Hugging Face Inference API](https://huggingface.co/docs/api-inference/index).

# ft-lora
Fine‑tuned LoRA models built on open‑source foundation LLMs, trained primarily with the FineTome‑100k dataset.

In this repository, a family of fine‑tuned large language models (LLMs) is created by applying LoRA to open‑source foundation models (e.g. Llama‑3.2, Phi‑4, etc.) on the publicly available dataset FineTome‑100k.  
The goal is to produce lighter, more efficient, and instruction‑tuned models that can be easily run on a CPU, while preserving strong performance on instruction following, dialogue, and general reasoning tasks.

What this repo contains:

- Training scripts / notebooks used to fine‑tune the models (`train.ipynb`, etc.)  
- Example inference / chat UIs using Gradio (`chat_ui/`) so you can easily try the models.  

## Contents

| Directory / File | Description |
|------------------|-------------|
| `train.ipynb` | Notebook showing how to fine‑tune a base LLama model with LoRA on FineTome‑100k. |
| `Phi_4_Conversational.ipynb` | Notebook showing how to fine‑tune a base Phi model with LoRA on FineTome‑100k. |
| `requirements.txt` | List of dependencies needed for training or inference. |
| `chat_ui/` | Example Gradio-based simple chat UIs to run the fine‑tuned model. The two available UIs is a vanilla chatbot UI and an agent with the capability of searching the web in order to answer questions. |
---

## HuggingFace

All the different fine-tuned models and spaces for interacting with the models can be found on this page: https://huggingface.co/ft-lora, where the best model so far can be tested here: https://huggingface.co/spaces/ft-lora/llama-agent-ui. 


## Improving Model Performance

Further experimentation in case of more available time and compute could provide improved model performance, both in terms of loss and perpexity (quantitative metrics), but also qualitative metrics in terms of conversational quality. These approaches can be broadly categorized into **model-centric** and **data-centric** improvements, and have to be weighted with need for latency of the chat UI. 

### 1. Model-Centric Approach

- **Hyperparameter tuning**: Experiment with learning rate, batch size, gradient accumulation, number of fine-tuning steps, or warmup schedules to improve convergence and stability.

    For example, a light warmup stabilizes early training, and the amount of max steps / epochs should be kept down or combined with early stopping for the smaller models, such as the 1B LLama model, which showed signs of overfitting in some cases. 

- **LoRA configuration**: Adjust the rank (`r`), alpha, and dropout values in LoRA adapters to achieve better balance between efficiency and expressivity.
- **Model architecture selection**: Consider using a larger base model or a different variant (e.g., Llama-3.3, or 8B foundation models) for more expressive power. Unfortunately, in this assignment we had a limited amount of GPU available and also prioritized low latency in the user interface, which is why larger models were disregarded.
- **Quantization strategies**: Explore different quantization levels (e.g. 8-bit instead of 4-bit) and offloading to optimize memory usage while minimizing performance degradation.

    8-bit quantization would probably have a better performance than the 4-bit version in this assignment, but be slower to use.

- **Regularization and optimization**: More weight decay or an early stopping algorithm could help stabilize training and improve generalization (reduce overfitting).

    A hypothesis is that increased regularization, like a weight decay of 0.04 instead of 0.01, would benefit the smaller models (especially the 1B model which we in https://huggingface.co/spaces/ft-lora/LLama-1B-UI could see have a bad performance, likely due to overfitting). 

### 2. Data-Centric Approach

Data-centric improvements could focus on improving the dataset, either in size, quality or diversity:

- **Expanding the dataset**: Other open-source instruction-following and dialogue datasets explored was ShareGPT90K and ultrachat_200k, that provide high-quality, diverse examples. Due to limited compute, however, these could not be properly explored.

- **Data cleaning and filtering**: Remove noisy, irrelevant, or duplicated examples from the dataset to improve signal-to-noise ratio during training. mlabonne/FineTome-100k does contain some duplicates in questions, like "Explain the concept of recursion with an example", which could harm generalizability since the model will be better at these types of questions and less diverse in its knowledge base.

- **Synthetic data augmentation**: Generate additional examples using existing models to create diverse instruction-following scenarios.

- **Domain-specific data**: Incorporate specialized datasets, such as legal, medical, or scientific texts to improve the model performance in those areas.

If enough time was given, both data-centric and model-centric approaches would be tried. 

## Findings

The 3B parameter LLama model with 4-bit quantization and 60 optimization steps was found to have the best tradeoff between accuracy and latency in this experiment. When entering a prompt ```"Explain how a blackhole forms."```, that was an existing human input from the FineTome dataset, this 3B model took about 35 seconds to generate an answer, and an excerpt of the answer is: 

```
Black holes! One of the most fascinating and mysterious objects in the universe.

A black hole forms when a massive star collapses in on itself and its gravity becomes so strong that not even light can escape. Here's a step-by-step explanation of how a black hole forms:

Step 1: A Massive Star Runs Out of Fuel
A massive star, typically with a mass at least 3-4 times that of the sun, exhausts its fuel sources (hydrogen and helium) in its core.

Step 2: Core Collapse
The star's core contracts under its own gravity, causing a massive amount of matter to be compressed into an incredibly small space. This compression causes a massive release of energy, known as a supernova explosion.

Step 3: Supernova Explosion
The supernova explosion is so powerful that it expels the star's outer layers into space, leaving behind the core. This core is the most massive object that remains.

Step 4: Core Collapses
The core continues to contract, and its gravity becomes so strong that it warps the fabric of spacetime around it. This creates a boundary called the event horizon, which marks the point of no return around a black hole.

Step 5: Singularity Formation
At the center of the core lies a singularity, a point of infinite density and zero volume, where the laws of physics as we know them break down.
...
```
... and so on

The 1B model answered after only 11 seconds to the same question, and the reply seemed coherent enough at first, but quickly started making nonsense:

```
A black hole forms when matter is pushed, often with gravity, causing the material to be squeezed. This squeezing is like a magnet that attracts matter. The force acting on matter is called pressure, and the force of gravity that's pulling on that matter is called pressure and pressure.

The more matter in a place, the more force the material will have when it is pulled in a direction. This is what makes the material of a black hole, as a black hole is a black hole, or a black hole, or a black hole.

The black hole then goes back to the black hole, where it's getting pulled, so to be a black hole, you need to be pulled in the direction of the black hole, which you can't do unless you're a black hole too, because that's how we do it.

The black hole then goes back to the black hole, where it's getting pulled, so to be a black hole too, you need to be pulled in the direction of the black hole, which you can't do unless you're a black hole too, because that's how we do it.

I like the word "black" so much, this word "black" means black. I know that a black black hole would mean a black hole would mean a black hole that is a black hole, a black hole, a black hole, a black hole, a black hole, a black hole, a black hole, a black hole, a black hole, a black hole, a black hole, a black hole, a black hole, a black hole, a black hole, a black hole, a black hole, a black hole, a black hole, a black hole, a black hole, 
...
```
... and so on

This shows that 1B models cannot represent as many complex relationships between concepts as 3B models. Over longer generations, errors like hallucinations, repetition and topic drift are present.

The bigger model showed slower decay of performance over long generations, while the smaller model showed a faster decay, even if context length is the same.

Although, the 3B model seems to give correct answers for questions present in the FineTome dataset such as the black hole question above, as well as for the question "Demonstrate how to use the XOR operator in Python", it would give better and more reliable responses if it could use external sources like documents or web search before it answers factual questions.

For that reason, we also tried making an agent (based on the 3B LLama model) with the capability to search the web when appropriate (via the DuckDuckGo API), and displayed the the results in this UI: https://huggingface.co/spaces/ft-lora/llama-agent-ui.


### LLama agent UI

In this UI, DuckDuckGo's instant answer API is used as a tool to fetch search results. The function ```duckduckgo_search``` returns a summarized version: the abstract plus a few related topics to the question. This allows the LLM to get up-to-date information from the web.

The 3B GGUF model still has ```n_ctx=1024``` for faster CPU performance, and ```use_mmap=True``` to save RAM, but instead of the standard ```respond``` function, an ```agent_respond``` function prompts the LLM to decide if a search is needed, and 'yes', it combines the user question + observation (search results) + conversation history to generate the final answer. For this model, we also changed the system prompt to "You are a helpful Chatbot who will help the user find the right answers." instead of "You are a friendly Chatbot.", which changed the tone of the model a bit.

Running this UI, takes a bit longer, as the question ```"Explain how a blackhole forms."``` could cause the model to take around 50 seconds to answer. For this use case, a larger 'max new tokens' was needed than for the 'vanilla' chatbot, since some tokens are used for taking a decision (not shown in the output). By asking ```"Explain how a blackhole forms."``` in the UI space: https://huggingface.co/spaces/ft-lora/llama-agent-ui, and setting the slider "Max new tokens" to 512, the answer from the model was: 

```
A black hole forms when a massive star collapses in on itself and its gravity becomes so strong that not even light can escape.

Here's a step-by-step explanation:

1. Massive star runs out of fuel: A massive star (typically 3-4 times the size of the sun) exhausts its fuel and dies.
2. Core collapses: The core of the star collapses under its own gravity, causing a massive amount of matter to be compressed into an extremely small space.
3. Event horizon forms: The point of no return, called the event horizon, is formed when the gravity of the collapsed core becomes so strong that any matter or radiation that crosses it will be trapped by the black hole.
4. Singularity forms: At the center of the black hole, a singularity forms, where the density and gravity are infinite.
5. Black hole is born: The combination of the collapsed core, event horizon, and singularity creates a black hole, a region from which nothing, including light, can escape.

This process typically occurs in massive stars, which can be found in galaxies. When a massive star undergoes a supernova explosion, the remnants of the star can sometimes collapse into a black hole.

I hope this explanation helps!
```

This explanation is quite similar to the one of the vanilla LLama chatbot, but it is backed up by a web search while still remaining helpful. However, it does make the UI slower.


