# Search-Space Specification

Phase 2 search spaces are versioned JSON objects with an environment identifier/version and a bounded `parameters` mapping. Each parameter is `float`, `integer`, `categorical`, or `fixed`. Float and integer bounds are inclusive and finite; categorical choices must be non-empty and unique.

The pump reference space is [configs/search/pump_stress.json](../configs/search/pump_stress.json). It searches demand, pump capacity, efficiency, initial level, target level, and leakage while fixing the integration step, horizon, and operating boundaries. The environment adapter performs the final physical compatibility validation.

The LunarLander reference space is [configs/search/lunarlander_risk.json](../configs/search/lunarlander_risk.json). It intentionally has no physical parameter dimensions because the stock adapter does not expose supported runtime gravity, wind, thrust, or sensor-fault configuration. It still provides a valid seed-policy search configuration for the observation-derived risk proxy.

Candidate generation is bounded and deterministic for a seeded random strategy. Invalid candidates are recorded as `invalid` and do not become physical failure discoveries.
