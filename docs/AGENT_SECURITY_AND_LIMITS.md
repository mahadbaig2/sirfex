# Agent security and limits

The optional Groq client reads `GROQ_API_KEY` from the process environment, uses one pinned model, sets request timeout and zero SDK retries, and sends no key, local path, checkpoint binary, raw trajectory, final partition, or arbitrary repository text in its prompt. Groq's best-effort JSON-schema output is untrusted and parsed into a strict versioned proposal before deterministic validation, including exact matches to catalogued evidence claims. The agent has no tool access, command execution, source editing, or access to regression thresholds.

When the process environment has no key, the CLI also reads `GROQ_API_KEY` from a small `.env` file in the current working directory. Process variables take precedence. `.env` is ignored by Git; never add it to a commit or public evidence export.

At most two proposal calls are permitted per run. Input and output token totals, latency, request ID, model, prompt hash, estimated cost, and provider failure type are recorded. Published token pricing may change; the cost is an estimate, not an invoice. A missing key, timeout, invalid response, or rejection produces no training. A provider failure after a completed training run must not invalidate its deterministic measurements.

The local stores are intended for a trusted workstation. Existing PPO checkpoint loading uses trusted locally produced files; do not load an untrusted checkpoint. Approval is a narrow CLI flag attached to a validated configuration hash and training cap. Automatically resuming a partially completed PPO run is not implemented. The offline mock is only a software test and cannot support any claim about GPT-OSS reasoning quality.
