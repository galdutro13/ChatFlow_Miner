import pytest


class StubProcessModelView:
    """A lightweight stub to stand in for ProcessModelView.

    - compute() returns `compute_return` or raises `compute_exc` if provided.
    - to_graphviz(**kwargs) returns `graphviz_return` or raises `graphviz_exc`.
      It records the last kwargs received in `last_to_graphviz_kwargs`.
    """

    def __init__(
        self,
        *,
        compute_return=None,
        compute_exc: Exception | None = None,
        graphviz_return=None,
        graphviz_exc: Exception | None = None,
    ) -> None:
        self._compute_return = compute_return
        self._compute_exc = compute_exc
        self._graphviz_return = graphviz_return
        self._graphviz_exc = graphviz_exc
        self.last_to_graphviz_kwargs: dict | None = None

    def compute(self):
        if self._compute_exc is not None:
            raise self._compute_exc
        return self._compute_return

    def to_graphviz(self, **kwargs):
        self.last_to_graphviz_kwargs = dict(kwargs)
        if self._graphviz_exc is not None:
            raise self._graphviz_exc
        return self._graphviz_return


@pytest.fixture
def stub_pmv_cls():
    return StubProcessModelView


@pytest.fixture
def make_stub_view(stub_pmv_cls):
    def _factory(
        *,
        compute_return=None,
        compute_exc: Exception | None = None,
        graphviz_return=None,
        graphviz_exc: Exception | None = None,
    ):
        return stub_pmv_cls(
            compute_return=compute_return,
            compute_exc=compute_exc,
            graphviz_return=graphviz_return,
            graphviz_exc=graphviz_exc,
        )

    return _factory


@pytest.fixture
def patch_registry_process_model_view(monkeypatch, stub_pmv_cls):
    # Redirect isinstance checks in the registry to our stub class
    import chatflow_miner.lib.process_models.model_registry as mr

    monkeypatch.setattr(mr, "ProcessModelView", stub_pmv_cls, raising=True)
    return stub_pmv_cls


@pytest.fixture
def registry(patch_registry_process_model_view):
    from chatflow_miner.lib.process_models.model_registry import ProcessModelRegistry

    return ProcessModelRegistry()


@pytest.fixture
def snap_registry(patch_registry_process_model_view):
    from chatflow_miner.lib.process_models.model_registry import ProcessModelRegistry

    return ProcessModelRegistry(cache_snapshots=True)


