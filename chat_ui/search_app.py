import subprocess
import gradio as gr
from huggingface_hub import hf_hub_download

# Install llama_cpp_python in the Space
subprocess.run("pip install llama_cpp_python==0.3.1", shell=True)
from llama_cpp import Llama

subprocess.run("pip install requests", shell=True)
import requests


def duckduckgo_search(query, max_results=3):
    """
    Perform a DuckDuckGo search and return summarized results.
    """
    url = "https://api.duckduckgo.com/"
    params = {
        "q": query,
        "format": "json",
        "no_redirect": 1,
        "skip_disambig": 1
    }
    
    try:
        resp = requests.get(url, params=params)
        data = resp.json()
        
        results = []
        # Add AbstractText if available for the source
        if data.get("AbstractText"):
            results.append(data["AbstractText"])
        
        # Related topics sometimes have extra info
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if "Text" in topic:
                results.append(topic["Text"])
            elif "Topics" in topic:
                for subtopic in topic["Topics"][:max_results]:
                    results.append(subtopic.get("Text", ""))
        
        return "\n".join(results) if results else "No relevant results found."
    
    except Exception as e:
        return f"Error fetching search results: {e}"

def search_web(query):
    """Perform a web search and return summarized results."""
    return duckduckgo_search(query)


# Download 3B GGUF model into HF Space storage
model_path = hf_hub_download(
    repo_id="ft-lora/llama3.2-3b-gguf-q4km",
    filename="llama3.2-3b-instruct-finetuned.gguf"
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

def agent_respond(question, history, system_message, max_tokens, temperature=0.7, top_p=0.95):
    messages = [{"role": "system", "content": system_message}] + history

    prompt = (
        f"Question: {question}\n"
        "You are an AI assistant that can use the tool `search_web(query)` to get up-to-date information.\n"
        "Decide if you need to search the web to answer this question.\n"
        "Respond with only 'Yes' or 'No'.\n"
        "Action:"
    )

    action_response = ""
    for chunk in llm.create_chat_completion(
        messages=messages + [{"role": "user", "content": prompt}],
        max_tokens=32,  # small for decision
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        delta = chunk["choices"][0]["delta"]
        token = delta.get("content", "")
        action_response += token

    # Search if appropriate
    if "yes" in action_response.lower():
        search_results = search_web(question)
        observation = f"Observation: {search_results}\nAnswer:"
    else:
        observation = "Answer:"

    # Ask model to generate final answer
    final_response = ""
    for chunk in llm.create_chat_completion(
        messages=messages + [{"role": "user", "content": f"{question}\n{observation}"}],
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        delta = chunk["choices"][0]["delta"]
        token = delta.get("content", "")
        final_response += token
        yield final_response


chatbot = gr.ChatInterface(
    agent_respond,
    type="messages",
    additional_inputs=[
        gr.Textbox(value="You are a helpful Chatbot who will help the user find the right answers.", label="System message"),
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
