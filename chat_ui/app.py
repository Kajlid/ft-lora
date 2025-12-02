import subprocess
import gradio as gr
from huggingface_hub import hf_hub_download

subprocess.run("pip install llama_cpp_python==0.3.1", shell=True)
from llama_cpp import Llama

# Download GGUF model into HF Space storage
model_path = hf_hub_download(
    repo_id="ft-lora/llama3.2-3b-gguf-q4km",
    filename="llama3.2-3b-instruct-finetuned.gguf"
)


llm = Llama(
	model_path=model_path,
    n_ctx=2048,
    use_mmap=True,    # use memory-mapped file to load a model
    chat_format="llama-3",
)


def respond(message, history, system_message, max_tokens, temperature, top_p):
    messages = [{"role": "system", "content": system_message}]
    
    for conv in history:
        messages.append(conv)      # add historical converational turns into history

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
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
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
