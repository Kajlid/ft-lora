import gradio as gr
from huggingface_hub import InferenceClient
import subprocess
from huggingface_hub import hf_hub_download

subprocess.run("pip install -V llama_cpp_python==0.3.1", shell=True)
# Download GGUF model into HF Space storage

from llama_cpp import Llama

model_path = hf_hub_download(
    repo_id="ft-lora/llama3.2-3b-gguf-q4km",
    filename="llama3.2-3b-instruct-finetuned.gguf"
)


llm = Llama(
	model_path=model_path,
    n_ctx=2048,
)


def respond(
    message,
    history: list[dict[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
    hf_token: gr.OAuthToken,
):
    """
    For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
    """
    # Model: the HuggingFace Transformers model saved during the last iteration of training
    # client = InferenceClient(token=hf_token.token, model="ft-lora/llama3.2-3b-instruct-finetuned")

    messages = [{"role": "system", "content": system_message}]
    
    for human, bot in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": bot})

    messages.append({"role": "user", "content": message})
    response = ""

    for message in llm.create_chat_completion(   
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        choices = message.choices
        token = ""
        if len(choices) and choices[0].delta.content:
            token = choices[0].delta.content

        response += token
        yield response


"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
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

with gr.Blocks() as demo:
    with gr.Sidebar():
        gr.LoginButton()
    chatbot.render()


if __name__ == "__main__":
    demo.launch()
