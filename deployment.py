# Production Inference Server using vLLM
# Run with: uvicorn deployment:app --host 0.0.0.0 --port 8000

from vllm import LLM, SamplingParams
from fastapi import FastAPI
import uvicorn
import os

app = FastAPI(title="WallStreet Agent API", version="1.0")

# Configuration
MODEL_PATH = "./WallStreet_Llama_GGUF.gguf" # Path to quantized model

print(f"🚀 Initializing Inference Engine from {MODEL_PATH}...")

# Initialize vLLM (High-performance inference engine)
# quantization="gguf" reduces memory usage by 60%
llm = LLM(model=MODEL_PATH, quantization="gguf", dtype="half")

@app.post("/v1/generate")
async def generate_advice(query: str, max_tokens: int = 250):
    """
    API Endpoint for Financial Advice Generation.
    Inputs:
        query: User's financial question
        max_tokens: Limit for response length
    """
    # Sampling params tuned for financial accuracy (low temp = less hallucination)
    sampling_params = SamplingParams(temperature=0.3, max_tokens=max_tokens)

    # Async Inference
    outputs = llm.generate([query], sampling_params)
    generated_text = outputs[0].outputs[0].text

    return {
        "status": "success",
        "query": query,
        "advice": generated_text,
        "engine": "Llama-3-8B-FinTuned"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
