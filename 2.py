import time
import psutil
import ollama
import pynvml

prompts = [
    "Hello!",
    "Tell me a joke.",
    "Summarize Romeo and Juliet.",
    "What's the capital of Japan?",
    "Give me a quick Python tip."
]

client = ollama.Client()

# --- CPU Run (full model on CPU) ---
CPU_TDP_W = 65  # AMD Ryzen 5 5600X
cpu_energy_joules = 0
print("=== CPU Responses ===")
for p in prompts:
    start_time = time.time()
    response = client.chat(model="llama3_1_8b_cpu", messages=[{"role": "user", "content": p}])
    elapsed = time.time() - start_time

    # Measure actual CPU utilization over the duration
    cpu_percent = psutil.cpu_percent(interval=None) / 100
    cpu_energy_joules += CPU_TDP_W * cpu_percent * elapsed

    print(f"Prompt: {p}\nTime: {elapsed:.2f} sec\n")

cpu_energy_kwh = cpu_energy_joules / 3600 / 1000
cpu_co2_kg = cpu_energy_kwh * 0.475
print(f"CPU Energy: {cpu_energy_kwh:.6f} kWh, CO2: {cpu_co2_kg:.6f} kg\n")

# --- GPU Run (full model on GPU) ---
pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
gpu_energy_joules = 0
print("=== GPU Responses ===")
for p in prompts:
    start_time = time.time()
    response = client.chat(model="llama3_1_8b_gpu", messages=[{"role": "user", "content": p}])
    elapsed = time.time() - start_time

    # Sample GPU power multiple times
    samples = 5
    sample_interval = elapsed / samples
    for _ in range(samples):
        power_mw = pynvml.nvmlDeviceGetPowerUsage(gpu_handle)
        gpu_energy_joules += power_mw / 1000 * sample_interval
        time.sleep(sample_interval)

    print(f"Prompt: {p}\nTime: {elapsed:.2f} sec\n")

gpu_energy_kwh = gpu_energy_joules / 3600
gpu_co2_kg = gpu_energy_kwh * 0.475
print(f"GPU Energy: {gpu_energy_kwh:.6f} kWh, CO2: {gpu_co2_kg:.6f} kg")

pynvml.nvmlShutdown()
