from codecarbon import EmissionsTracker
import ollama
import time

# Your prompts
prompts = [
    "Hello!",
    "Tell me a joke.",
    "Summarize Romeo and Juliet.",
    "What's the capital of Japan?",
    "Give me a quick Python tip."
]

# Initialize Ollama client
client = ollama.Client()

# --- CPU Run ---
cpu_tracker = EmissionsTracker(project_name="llama_cpu_run")
cpu_tracker.start()

print("=== CPU Responses ===")
for p in prompts:
    start = time.time()
    response = client.chat(model="llama3_1_8b_cpu", messages=[{"role": "user", "content": p}])
    end = time.time()
    print(f"Prompt: {p}\nResponse: {response}\nTime: {end-start:.2f} sec\n")

cpu_emissions = cpu_tracker.stop()
print(f"Estimated CPU emissions: {cpu_emissions:.6f} kg CO2\n")

# --- GPU Run ---
gpu_tracker = EmissionsTracker(project_name="llama_gpu_run")
gpu_tracker.start()

print("=== GPU Responses ===")
for p in prompts:
    start = time.time()
    response = client.chat(model="llama3_1_8b_gpu", messages=[{"role": "user","content": p}])
    end = time.time()
    print(f"Prompt: {p}\nResponse: {response}\nTime: {end-start:.2f} sec\n")

gpu_emissions = gpu_tracker.stop()
print(f"Estimated GPU emissions: {gpu_emissions:.6f} kg CO2")
