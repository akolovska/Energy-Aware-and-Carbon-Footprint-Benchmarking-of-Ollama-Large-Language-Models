import time
import psutil
import ollama
import pynvml
import subprocess
import gc

# List of models to test
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

prompts = [
    "Hello!",
    "Tell me a joke.",
    "What's the capital of Japan?",
    "Tell me a short fact.",
    "Give a simple Python example for loops."
]

CPU_TDP_W = 65

# Initialize GPU monitoring
pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

for model in models:
    print(f"\n=== Running model: {model} ===")
    gc.collect()
    # === CLEAR GPU MEMORY ===
    if "-gpu" in model:
        print("Stopping Ollama to clear GPU memory...")
        subprocess.run(["ollama", "stop"])
        time.sleep(2)  # Give time for memory to clear
        subprocess.run(["ollama", "start"])
        time.sleep(2)  # Ensure Ollama restarted

    client = ollama.Client()

    energy_joules = 0
    start_time = time.time()

    for p in prompts:
        resp_start = time.time()
        client.chat(model=model, messages=[{"role": "user", "content": p}])
        elapsed = time.time() - resp_start

        if "-cpu" in model:
            cpu_percent = psutil.cpu_percent(interval=None) / 100
            energy_joules += CPU_TDP_W * cpu_percent * elapsed
        elif "-gpu" in model:
            samples = 5
            sample_interval = elapsed / samples
            for _ in range(samples):
                power_mw = pynvml.nvmlDeviceGetPowerUsage(gpu_handle)
                energy_joules += power_mw / 1000 * sample_interval
                time.sleep(sample_interval)

    elapsed_total = time.time() - start_time
    energy_kwh = energy_joules / 3600 / 1000
    co2_kg = energy_kwh * 0.475

    print(f"Model: {model}")
    print(f"Total Time: {elapsed_total:.2f} sec")
    print(f"Energy: {energy_kwh:.6f} kWh")
    print(f"CO2: {co2_kg:.6f} kg")

pynvml.nvmlShutdown()
