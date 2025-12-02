import subprocess
import gradio as gr
from huggingface_hub import hf_hub_download

# Install llama_cpp_python in the Space
subprocess.run("pip install llama_cpp_python==0.3.1", shell=True)
from llama_cpp import Llama

# Download 1B GGUF model into HF Space storage
model_path = hf_hub_download(
    repo_id="ft-lora/llama3.2-1b-gguf-auto",           # 1B GGUF repo
    filename="llama3.2-1b-instruct-finetuned.gguf"     # 1B GGUF file
)

# Initialize llama.cpp with smaller context & both CPU cores
llm = Llama(
    model_path=model_path,
    n_ctx=1024,        # smaller context -> faster on CPU
    n_threads=2,       # use both vCPUs on HF Spaces
    use_mmap=True,     # memory-mapped loading
    chat_format="llama-3",
)


def respond(message, history, system_message, max_tokens, temperature, top_p):
    messages = [{"role": "system", "content": system_message}]
    
    # history is already a list of {role, content} dicts
    for conv in history:
        messages.append(conv)

    messages.append({"role": "user", "content": message})
    response = ""

    for chunk in llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        delta = chunk["choices"][0]["delta"]
        token = delta.get("content", "")
        response += token
        yield response
           

chatbot = gr.ChatInterface(
    respond,
    type="messages",
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        # Smaller default generation length for faster replies
        gr.Slider(minimum=1, maximum=512, value=128, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
)

demo = gr.Blocks()
with demo:
    chatbot.render()


if __name__ == "__main__":
    demo.launch()
