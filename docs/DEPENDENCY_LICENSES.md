# Dependency License Notes

SIRFEX is licensed under Apache-2.0. That license applies to SIRFEX code only; dependencies retain their own licenses and must be redistributed according to their terms.

The Phase 1 environment records installed dependency versions in each manifest. The current dependency families and their package metadata/license notices are:

| Dependency | Use | License/notice obligation |
|---|---|---|
| NumPy | Numerical arrays | Preserve the package's BSD license and notices when redistributing it. |
| Gymnasium | Environment API and LunarLander | Preserve its MIT license and notices. |
| Box2D | LunarLander physics binding | Preserve its zlib license and notices. |
| pygame-ce | Optional headless/rendering support dependency | Preserve its LGPL-2.1 license and notices. |
| PyTorch | Optional PPO smoke test | Preserve the package's bundled license and notices. |
| Stable-Baselines3 | Optional PPO smoke test | Preserve its MIT license and notices. |
| pytest/Ruff | Development and CI tooling | Preserve their package licenses when redistributed. |

This is an engineering inventory, not legal advice. Before publishing bundled third-party binaries or a packaged distribution, review the exact installed versions' license files and notices.
