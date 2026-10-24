# Scenario Specification

SIRFEX Phase 1 scenarios are JSON objects with schema version `1`:

```json
{
  "environment_id": "pump",
  "environment_version": "sirfex-pump:1",
  "schema_version": "1",
  "parameters": {},
  "seed": 0,
  "description": "Nominal pump baseline scenario",
  "tags": ["baseline"],
  "provenance": {"source": "local"}
}
```

`environment_id`, `parameters`, and `seed` are required. Environment versions are checked when supplied. Scenario values are JSON-serializable and are not silently coerced.

## Pump parameters

The pump adapter exposes `dt` (positive seconds), `horizon` (positive integer steps), `target_level`, `initial_level`, `demand`, `efficiency`, `min_level`, and `max_level` in the normalized physical ranges documented by `sirfex scenarios list --env pump`. `pump_capacity` and `leak` are nonnegative normalized flow parameters. Initial and target levels must lie within the ordered constraint boundaries.

## LunarLander parameters

The stock Gymnasium adapter exposes no runtime scenario parameters. Gravity, wind, thrust degradation, and sensor faults are not advertised because they are not implemented by this adapter.

Scenarios can be loaded and validated with:

```powershell
sirfex scenarios validate --file configs/pump_baseline.json
```
