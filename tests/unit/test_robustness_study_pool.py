"""Controller resource accounting survives session eviction."""

from sirfex.robustness.study_execute import ControllerPool


def test_controller_seconds_accumulate_across_eviction(monkeypatch):
    import sirfex.robustness.study_execute as module

    class Session:
        def __init__(self, spec):
            self.seconds = 0.0

        def evaluate(self, scenario):
            self.seconds += 1.0
            return scenario

        def close(self):
            pass

    monkeypatch.setattr(module, "_Session", Session)
    pool = ControllerPool({name: {"id": name} for name in ("a", "b", "c")})
    for name in ("a", "b", "c", "a"):
        pool.evaluate(name, "synthetic")
    assert pool.seconds("a") == 2.0
    assert pool.seconds("b") == 1.0
    assert pool.seconds("c") == 1.0
    pool.close()
    assert pool.seconds("a") == 2.0
