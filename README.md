### LLM Results Evalution

Refine messy voice dictation into clean, structured text using a local LLM (llama.cpp) and validate it with deepeval.

This repo contains a minimal harness that:
- calls your local model via a LocalModel.generate() method
- checks the output against an expected target with a simple Correctness metric.

If the model’s output doesn’t match, the test fails. No sugar-coating.
