from codecarbon import EmissionsTracker
import ollama
import time

models = [
    "qwen2-7b-cpu", "qwen2-7b-gpu",
    "openthinker-32b-cpu", "openthinker-32b-gpu",
    "qwq-32b-cpu", "qwq-32b-gpu",
    "qwen-32b-cpu", "qwen-32b-gpu",
    "mistral-small-24b-cpu", "mistral-small-24b-gpu",
    "deepseek-coder-33b-cpu", "deepseek-coder-33b-gpu",
    "gemma2-27b-cpu", "gemma2-27b-gpu",
    "qwen3-30b-cpu", "qwen3-30b-gpu",
    "qwen2-5-32b-cpu", "qwen2-5-32b-gpu",
    "deepseek-r1-32b-cpu", "deepseek-r1-32b-gpu",
    "gemma3-27b-cpu", "gemma3-27b-gpu",
    "orca-mini-13b-cpu", "orca-mini-13b-gpu",
    "olmo2-13b-cpu", "olmo2-13b-gpu",
    "llama2-13b-cpu", "llama2-13b-gpu",
    "starcoder2-15b-cpu", "starcoder2-15b-gpu",
    "mistral-nemo-12b-cpu", "mistral-nemo-12b-gpu",
    "llava-13b-cpu", "llava-13b-gpu",
    "phi3-14b-cpu", "phi3-14b-gpu",
    "phi4-14b-cpu", "phi4-14b-gpu",
    "gpt-oss-20b-cpu", "gpt-oss-20b-gpu",
    "llama3-2-vision-11b-cpu", "llama3-2-vision-11b-gpu",
    "llava-llama3-8b-cpu", "llava-llama3-8b-gpu",
    "codegemma-7b-cpu", "codegemma-7b-gpu",
    "gemma-7b-cpu", "gemma-7b-gpu",
    "qwen2-7b-cpu", "qwen2-7b-gpu",
    "dolphin3-8b-cpu", "dolphin3-8b-gpu",
    "llama3-8b-cpu", "llama3-8b-gpu",
    "llama3-1-8b-cpu", "llama3-1-8b-gpu",
    "deepseek-r1-8b-cpu", "deepseek-r1-8b-gpu",
    "qwen3-8b-cpu", "qwen3-8b-gpu",
    "mistral-7b-cpu", "mistral-7b-gpu"
]

# Your prompts
prompts = [
    "Hello!",
    "Tell me a joke.",
    "What's the capital of Japan?",
    "Tell me a short fact.",
    "Give a simple Python example for loops."
]
client = ollama.Client
# Iterate through all models
for model in models:
    print(f"\n=== Running model: {model} ===")

    tracker = EmissionsTracker(project_name=model)
    tracker.start()

    model_start = time.time()
    for p in prompts:
        try:
            response = ollama.chat(model=model, messages=[{"role": "user", "content": p}])
        except Exception as e:
            print(f"Error running {model}: {e}")
    model_end = time.time()

    emissions = tracker.stop()
    total_time = model_end - model_start
    print(f"Model: {model} | Total time: {total_time:.2f} sec | Emissions: {emissions:.6f} kg CO2")
