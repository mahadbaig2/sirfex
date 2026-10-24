# SIRFEX

Simulation-based Integrity, Robustness & Failure Exploration

SIRFEX is an open-source, local-first Physical AI testing engine for autonomous controllers. It is designed to evaluate controllers in simulated physical environments, discover reproducible failure scenarios, and help engineers measure and improve controller robustness.

> Find failure conditions before they find your controller.

## Project status

SIRFEX is at the repository-initialization stage. The technical requirements and architecture are defined in [the technical PRD](SIRFEX_Technical_PRD_and_Architecture_v1.md); Phase 0 implementation and hardware feasibility benchmarking are planned next.

The project is intentionally CPU-first and local-first. It will not require paid APIs, hosted simulation, cloud training, or a dedicated GPU.

## Planned v1 scope

- Headless LunarLander and simplified industrial pump-control environments.
- Shared environment, controller, and evaluation interfaces.
- Reproducible scenario execution, trajectory evidence, and failure replay.
- CPU benchmarking and optional PPO workflows.
- A Python package and CLI using the `sirfex` namespace and command.

Planned capabilities are not presented as implemented features. See the PRD for the complete requirements, constraints, architecture, and roadmap.

## Development

Development is being carried out incrementally in public. Each milestone will be verified locally before it is pushed, and this README plus the GitHub repository description will be kept aligned with the actual implementation status.

The project is intended to support Python environments compatible with Gymnasium, Box2D, NumPy, PyTorch, and Stable-Baselines3 as implementation proceeds. Installation and CLI commands will be added when the corresponding Phase 0 functionality exists.

## License

The project license has not yet been selected. MIT and Apache-2.0 are under consideration; the choice will be documented before licensing the repository.

## History

SIRFEX was formerly called AEGIS. The rename is recorded in [CHANGELOG.md](CHANGELOG.md).
