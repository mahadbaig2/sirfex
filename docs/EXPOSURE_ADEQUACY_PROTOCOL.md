# Exposure adequacy protocol

The frozen Phase 3.5 gate for each stress-training arm requires at least 10 completed stress episodes, 1,000 actual stress interactions and three distinct discoveries from the seven-scenario pool. It also requires valid assignments, no interrupted resets and exact reconciliation to PPO interactions. `adequate_exposure` means all checks pass; `insufficient_exposure` means valid measured exposure fell short; `invalid_exposure` means missing, incomplete or inconsistent evidence.

The gate determines whether a curriculum intervention is interpretable. It does not classify physical safety or controller regression. A candidate must pass the gate on both development seeds before policy outcomes can qualify it for confirmation. The threshold was frozen in [the preregistration](evidence/phase_3_5_preregistered.json) before screening. These are study design thresholds, not universal scientific constants.
