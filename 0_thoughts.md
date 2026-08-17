

notes
  - Best open models run ~128k–262k vocab tokens (Llama 3: 128,256; DeepSeek-V3: ~128k; Qwen2.5/3: ~152k; Mistral Tekken: 131,072; Gemma 2/3: 256k/262,144; gpt-oss: ~201k), vs. the
  old 32k baseline — and config vocab_size is often padded above the true token count.