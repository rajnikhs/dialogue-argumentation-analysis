import subprocess
import time
import argparse
import os
import signal

import llm_annotation

VLLM_HOST = "127.0.0.1"
VLLM_PORT = 7997
VLLM_BASE_URL = f"http://{VLLM_HOST}:{VLLM_PORT}/v1"

MODELS = [
    {
        "name": "llama3",
        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
    }
]

VLLM_STARTUP_TIMEOUT = 600  # seconds to wait for vllm server to be ready

def start_vllm_server(model_id):
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_id,
        "--host", VLLM_HOST,
        "--port", str(VLLM_PORT),
        "--max_model_len", "4096",
        "--served-model-name", model_id,
	"--enforce-eager",
    ]
    proc = subprocess.Popen(cmd)
    print(f"Started vllm server (pid {proc.pid}) for {model_id}")
    return proc

def wait_for_server(timeout=VLLM_STARTUP_TIMEOUT):
    import urllib.request
    deadline = time.time() + timeout
    url = f"http://{VLLM_HOST}:{VLLM_PORT}/health"
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(3)
    return False

def stop_vllm_server(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    print(f"vllm server stopped")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input dataset CSV file path")
    parser.add_argument("output_path", help="Path to write final output CSV/JSONL file")
    args = parser.parse_args()

    for model in MODELS:
        name = model["name"]
        model_id = model["model_id"]

        print(f"\n=== [{name}] Starting vllm server for {model_id} ===")
        proc = start_vllm_server(model_id)
        try:
            if not wait_for_server():
                raise RuntimeError(f"vllm server did not start within {VLLM_STARTUP_TIMEOUT}s")

            print(f"[{name}] Running classification pipeline...")
            llm_annotation.run_annotation(model_id, VLLM_BASE_URL, args.input, args.output_path)
            print(f"[{name}] Classification complete.")

        finally:
            stop_vllm_server(proc)
