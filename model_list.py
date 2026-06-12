'''
This file contains the list of models for each backend.
'''

MODELS = {
    
    "openai": [
        "gpt-5-nano",
        "gpt-5.4-nano",
        "gpt-4.1-nano",
        "gpt-4o-mini",
        "gpt-4.1-mini",
        "gpt-5-mini",
        "gpt-5.4-mini",
        "gpt-3.5-turbo",
        "o4-mini",
        "o3-mini",
        "o1-mini",
    ],
    "ollama": ["tinyllama", "gemma2:2b", "llama3.2:1b-instruct", "gemma4"],
    "gemini": ["gemini-2.5-flash", "gemini-2.0-flash-lite"],
    "huggingface": ["meta-llama/Llama-3.2-1B-Instruct", "zai-org/GLM-5.1:fastest"],
    "nvidia_nim": ["moonshotai/kimi-k2.6", "meta/llama-3.1-8b-instruct"],
}