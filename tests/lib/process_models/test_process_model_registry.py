import pytest


# -------- Core mapping behavior --------


def test_mapping_basics_iteration_order_and_len_repr(registry, make_stub_view):
    """Keys iterate in insertion order; len and repr reflect keys in order."""
    v1 = make_stub_view(compute_return=1)
    v2 = make_stub_view(compute_return=2)

    registry.add("a", v1)
    registry.add("b", v2)

    assert list(iter(registry)) == ["a", "b"]  # insertion order
    assert len(registry) == 2
    assert registry["a"] is v1
    assert registry["b"] is v2

    r = repr(registry)
    # Must list keys in order inside brackets
    assert r.endswith("['a', 'b'])") or r.endswith("['a', 'b']")


def test_setitem_accepts_only_valid_view_and_rejects_none(registry, make_stub_view):
    """__setitem__ validates key and requires ProcessModelView; None rejected with message."""
    v = make_stub_view(compute_return=1)
    registry.add("a", v)

    with pytest.raises(TypeError) as ei:
        registry["x"] = None
    # exact message from implementation
    assert "Atribuir None não é permitido" in str(ei.value)

    with pytest.raises(TypeError):
        registry["x"] = object()  # not a ProcessModelView

    # valid setitem updates or inserts
    v2 = make_stub_view(compute_return=2)
    registry["x"] = v2
    assert registry["x"] is v2


@pytest.mark.parametrize("bad_name", [123, 1.2, (), [], None])
def test_add_rejects_non_str_names(registry, make_stub_view, bad_name):
    """add() rejects non-str names via _validate_name with TypeError."""
    with pytest.raises(TypeError):
        registry.add(bad_name, make_stub_view())  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_name", [""])
def test_add_rejects_empty_name(registry, make_stub_view, bad_name):
    """add() rejects empty str names with ValueError."""
    with pytest.raises(ValueError):
        registry.add(bad_name, make_stub_view())


# -------- Placeholder rules --------


def test_placeholder_only_first_insert_and_readd_same_single_with_overwrite(registry):
    """add(name, None) allowed only first; re-adding same when len==1 and overwrite=True allowed."""
    registry.add("p", None)
    assert registry["p"] is None

    # re-adding same placeholder with overwrite when only one item exists
    registry.add("p", None, overwrite=True)
    assert registry["p"] is None


def test_placeholder_otherwise_rejected_and_setitem_none_always_typeerror(registry, make_stub_view):
    """Any add(name,None) beyond first must raise ValueError; __setitem__(k,None) always TypeError."""
    registry.add("p", None)
    with pytest.raises(ValueError):
        registry.add("q", None)

    # after filling placeholder, still cannot add placeholder elsewhere
    registry["p"] = make_stub_view()
    with pytest.raises(ValueError):
        registry.add("r", None)

    with pytest.raises(TypeError):
        registry["x"] = None


# -------- add and overwrite semantics + cache invalidation --------


def test_add_and_overwrite_and_cache_invalidation(registry, make_stub_view):
    """add inserts; overwrite=False on existing raises; overwrite=True replaces and invalidates cached tuples."""
    v1 = make_stub_view(compute_return=1)
    v2 = make_stub_view(compute_return=2)
    registry.add("a", v1)

    # Snapshot copies are rebuilt by next call; check by identity change
    t1_names = registry.names_tuple()

    with pytest.raises(KeyError):
        registry.add("a", v2, overwrite=False)

    registry.add("a", v2, overwrite=True)

    assert registry["a"] is v2

    t2_names = registry.names_tuple()
    t2_vals = registry.values_tuple()
    # When cache_snapshots is False, tuples are recreated each call, but we still want to
    # ensure mutation invalidation path ran; contents equal and identities may differ.
    assert t2_names == t1_names == ("a",)
    assert t2_vals == (v2,)


# -------- add_many semantics --------


@pytest.mark.parametrize("as_mapping", [True, False])
def test_add_many_accepts_single_placeholder_only_when_empty(registry, as_mapping):
    """add_many allows exactly one (name,None) entry when empty; populates registry accordingly."""
    data = {"p": None}
    entries = data if as_mapping else list(data.items())
    registry.add_many(entries)
    assert registry["p"] is None


@pytest.mark.parametrize(
    "entries_builder",
    [
        lambda mv: {"a": mv(), "b": mv()},
        lambda mv: [("a", mv()), ("b", mv())],
    ],
)
def test_add_many_delegates_to_add_and_inserts(registry, make_stub_view, entries_builder):
    """When no placeholders, add_many adds all entries from mapping or iterable."""
    entries = entries_builder(make_stub_view)
    registry.add_many(entries)
    assert set(registry.names) == {"a", "b"}
    assert all(v is not None for v in registry.values_view)


@pytest.mark.parametrize(
    "entries",
    [
        {"a": None, "b": None},
        [("a", None), ("b", None)],
        {"a": None, "b": object()},
        [("a", None), ("b", object())],
    ],
)
def test_add_many_rejects_multiple_or_mixed_placeholders_or_non_empty(registry, entries):
    """Any batch with more than one None or mixed or non-empty registry must raise ValueError."""
    # non-empty + any None is invalid too
    with pytest.raises(ValueError):
        registry.add_many(entries)

    # Make non-empty and try again with a single placeholder
    registry.add("x", None)
    with pytest.raises(ValueError):
        registry.add_many({"p": None})


# -------- rename semantics --------


def test_rename_moves_to_end_and_overwrite_semantics(registry, make_stub_view):
    """rename moves entry to end; overwrite rules observed; same-name is no-op."""
    v1 = make_stub_view()
    v2 = make_stub_view()
    registry.add("a", v1)
    registry.add("b", v2)

    # same-name no-op
    registry.rename("a", "a")
    assert list(registry) == ["a", "b"]

    # move a -> c to end
    registry.rename("a", "c")
    assert list(registry) == ["b", "c"]

    # overwrite=False prevents replacing existing
    registry.add("x", v1)
    with pytest.raises(KeyError):
        registry.rename("c", "x", overwrite=False)

    # overwrite=True replaces existing and moves to end
    registry.rename("c", "x", overwrite=True)
    assert list(registry) == ["b", "x"]


# -------- remove semantics --------


def test_remove_returns_previous_value_and_deletes(registry, make_stub_view):
    """remove returns previous value (including None) and deletes the entry; missing raises KeyError."""
    registry.add("p", None)
    prev = registry.remove("p")
    assert prev is None
    assert len(registry) == 0

    registry.add("a", make_stub_view())
    val = registry.remove("a")
    assert hasattr(val, "compute")
    assert len(registry) == 0

    with pytest.raises(KeyError):
        registry.remove("missing")


# -------- get_many --------


@pytest.mark.parametrize("missing_mode", ["error", "skip", "none"])
def test_get_many_behaviors(registry, make_stub_view, missing_mode):
    """get_many supports error/skip/none for missing names and returns in positional order."""
    registry.add("a", make_stub_view())
    registry.add("b", make_stub_view())

    names = ["a", "missing", "b"]

    if missing_mode == "error":
        with pytest.raises(KeyError):
            registry.get_many(names, missing=missing_mode)
    elif missing_mode == "skip":
        out = registry.get_many(names, missing=missing_mode)
        assert len(out) == 2 and out[0] is not None and out[1] is not None
    else:
        out = registry.get_many(names, missing=missing_mode)
        assert out[0] is not None and out[1] is None and out[2] is not None

    with pytest.raises(ValueError):
        registry.get_many(["a"], missing="bad")


# -------- compute_map --------


@pytest.mark.parametrize("on_error", ["raise", "skip", "none"])
def test_compute_map_placeholder_and_exception_paths(registry, make_stub_view, on_error):
    """compute_map respects on_error for placeholders and compute exceptions; subset names pass-through."""
    ok = make_stub_view(compute_return=10)
    bad = make_stub_view(compute_exc=RuntimeError("boom"))
    registry.add("p", None)
    registry.add("ok", ok)
    registry.add("bad", bad)

    subset = ["ok", "bad"]

    if on_error == "raise":
        # placehoder triggers specific error
        with pytest.raises(ValueError):
            registry.compute_map()

        # subset: bad raises underlying
        with pytest.raises(RuntimeError):
            registry.compute_map(names=subset, on_error=on_error)
    elif on_error == "skip":
        res_all = registry.compute_map(on_error=on_error)
        assert set(res_all.keys()) == {"ok"}

        res_subset = registry.compute_map(names=subset, on_error=on_error)
        assert set(res_subset.keys()) == {"ok"}
    else:  # none
        res_all = registry.compute_map(on_error=on_error)
        assert res_all.get("p") is None and res_all.get("ok") == 10 and "bad" in res_all and res_all["bad"] is None

        res_subset = registry.compute_map(names=subset, on_error=on_error)
        assert res_subset == {"ok": 10, "bad": None}

    with pytest.raises(ValueError):
        registry.compute_map(on_error="invalid")


# -------- to_graphviz_map --------


@pytest.mark.parametrize("on_error", ["raise", "skip", "none"])
def test_to_graphviz_map_mirrors_compute_and_forwards_kwargs(registry, make_stub_view, on_error):
    """to_graphviz_map mirrors error handling and forwards kwargs to view.to_graphviz."""
    g_ok = object()
    ok = make_stub_view(graphviz_return=g_ok)
    bad = make_stub_view(graphviz_exc=RuntimeError("boom"))
    registry.add("p", None)
    registry.add("ok", ok)
    registry.add("bad", bad)

    kwargs = {"layout": "dot", "rankdir": "LR", "dpi": 100}

    if on_error == "raise":
        with pytest.raises(ValueError):
            registry.to_graphviz_map(**kwargs)
        with pytest.raises(RuntimeError):
            registry.to_graphviz_map(names=["ok", "bad"], on_error=on_error, **kwargs)
    elif on_error == "skip":
        out = registry.to_graphviz_map(on_error=on_error, **kwargs)
        assert set(out.keys()) == {"ok"}
        assert ok.last_to_graphviz_kwargs == kwargs
    else:
        out = registry.to_graphviz_map(on_error=on_error, **kwargs)
        assert out.get("p") is None and out.get("ok") is g_ok and out.get("bad") is None
        assert ok.last_to_graphviz_kwargs == kwargs

    with pytest.raises(ValueError):
        registry.to_graphviz_map(on_error="invalid")


# -------- Views vs snapshots --------


def test_dynamic_views_reflect_mutations(registry, make_stub_view):
    """names/values_view/items reflect live changes without copying."""
    v = make_stub_view()
    registry.add("a", v)
    names_view = registry.names
    values_view = registry.values_view
    items_view = registry.items()

    assert list(names_view) == ["a"]
    assert list(values_view)[0] is v
    assert list(items_view) == [("a", v)]

    registry.add("b", make_stub_view())
    assert list(names_view) == ["a", "b"]
    assert len(values_view) == 2
    assert list(items_view)[-1][0] == "b"


def test_list_snapshots_are_copies(registry, make_stub_view):
    """names_list/values_list return copies, not live views."""
    registry.add("a", make_stub_view())
    n1 = registry.names_list()
    v1 = registry.values_list()
    registry.add("b", make_stub_view())
    n2 = registry.names_list()
    v2 = registry.values_list()
    assert n1 == ["a"] and n2 == ["a", "b"]
    assert len(v1) == 1 and len(v2) == 2


def test_tuple_snapshots_without_cache_are_rebuilt_each_call(registry, make_stub_view):
    """When cache_snapshots=False, tuples reflect current contents and identity may differ after mutation."""
    registry.add("a", make_stub_view())
    t1n = registry.names_tuple()
    t1v = registry.values_tuple()
    registry.add("b", make_stub_view())
    t2n = registry.names_tuple()
    t2v = registry.values_tuple()
    assert t1n != t2n and t1v != t2v


def test_tuple_snapshots_with_cache_invalidate_on_mutations(snap_registry, make_stub_view):
    """When cache_snapshots=True, tuples are cached until any mutation; invalidate on set, delete, rename, clear."""
    r = snap_registry
    r.add("a", make_stub_view())
    r.add("b", make_stub_view())

    tn1 = r.names_tuple()
    tv1 = r.values_tuple()
    assert r.names_tuple() is tn1 and r.values_tuple() is tv1  # cached

    # setitem mutation
    r["c"] = make_stub_view()
    tn2 = r.names_tuple()
    tv2 = r.values_tuple()
    assert tn2 != tn1 and tv2 != tv1

    # delete mutation
    del r["a"]
    tn3 = r.names_tuple()
    assert tn3 != tn2

    # rename mutation
    r.rename("b", "z")
    tn4 = r.names_tuple()
    assert tn4 != tn3 and tn4[-1] == "z"

    # clear mutation
    r.clear()
    tn5 = r.names_tuple()
    assert tn5 == ()


# -------- freeze --------


def test_freeze_returns_mapping_proxy_and_reflects_mutations(registry, make_stub_view):
    """freeze returns read-only MappingProxyType; underlying mutations reflect in the proxy."""
    from types import MappingProxyType

    v = make_stub_view()
    registry.add("a", v)
    frozen = registry.freeze()
    assert isinstance(frozen, MappingProxyType)
    assert frozen["a"] is v

    # Attempt to mutate proxy should raise
    with pytest.raises(TypeError):
        frozen["x"] = v  # type: ignore[index]

    # Mutating registry is visible via proxy
    registry.add("b", make_stub_view())
    assert set(frozen.keys()) == {"a", "b"}


# -------- has_placeholder and clear --------


def test_has_placeholder_transitions_and_clear(registry, make_stub_view):
    """has_placeholder() toggles as entries added/filled/removed; clear() empties and invalidates caches."""
    assert registry.has_placeholder() is False
    registry.add("p", None)
    assert registry.has_placeholder() is True

    registry["p"] = make_stub_view()
    assert registry.has_placeholder() is False

    registry.add("q", make_stub_view())
    assert registry.has_placeholder() is False

    # remove and check
    registry.remove("q")
    assert registry.has_placeholder() is False

    # clear
    registry.clear()
    assert len(registry) == 0
    assert registry.names_tuple() == () and registry.values_tuple() == ()


