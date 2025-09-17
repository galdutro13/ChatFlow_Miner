from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterable, Iterator, ItemsView, KeysView, Mapping, MutableMapping, ValuesView
from types import MappingProxyType
from typing import Any, Literal, Optional

# If ProcessModelView is defined elsewhere, import it here:
from .view import ProcessModelView

@dataclass(slots=True)
class ProcessModelRegistry(MutableMapping[str, Optional[ProcessModelView]]):
    """
    Registry especializado para objetos ProcessModelView, indexados por nome único.

    Regras:
    - É permitido adicionar um nome sem view associada (placeholder) somente
      quando o registry está vazio (primeira inserção).
    - Após a primeira inserção, toda nova entrada deve ter uma view associada.
    - Você pode preencher o placeholder posteriormente via atribuição:
        registry[name] = view
      ou via add(name, view, overwrite=True).

    Características:
    - Operações O(1) (média) para get/set/delete.
    - Iteração determinística por ordem de inserção.
    - names e values expõem Views dinâmicas (sem cópia).
    - Snapshots opcionais com cache.
    """

    cache_snapshots: bool = False

    _data: dict[str, Optional[ProcessModelView]] = field(default_factory=dict, init=False, repr=False)
    _names_cache: tuple[str, ...] | None = field(default=None, init=False, repr=False)
    _values_cache: tuple[Optional[ProcessModelView], ...] | None = field(default=None, init=False, repr=False)

    # --- Core MutableMapping interface ---

    def __getitem__(self, key: str) -> Optional[ProcessModelView]:
        return self._data[key]

    def __setitem__(self, key: str, value: Optional[ProcessModelView]) -> None:
        self._validate_name(key)
        if value is None:
            # Bloqueia criação de placeholders via atribuição direta.
            raise TypeError(
                "Atribuir None não é permitido. Somente a primeira inserção via add(name) "
                "pode criar um placeholder."
            )
        self._validate_view(value)
        self._data[key] = value
        self._invalidate_cache()

    def __delitem__(self, key: str) -> None:
        del self._data[key]
        self._invalidate_cache()

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return f"{cls}({list(self._data.keys())!r})"

    # --- Views (sem cópia) ---

    @property
    def names(self) -> KeysView[str]:
        """View dinâmica dos nomes (keys)."""
        return self._data.keys()

    @property
    def values_view(self) -> ValuesView[Optional[ProcessModelView]]:
        """View dinâmica dos valores: ProcessModelView ou None (placeholder)."""
        return self._data.values()

    def items(self) -> ItemsView[str, Optional[ProcessModelView]]:
        """View dinâmica de (nome, view|None) (sem cópia)."""
        return self._data.items()

    # --- Snapshots (com cópia, opcionais com cache) ---

    def names_list(self) -> list[str]:
        """Lista materializada de nomes."""
        return list(self._data.keys())

    def values_list(self) -> list[Optional[ProcessModelView]]:
        """Lista materializada de valores: ProcessModelView ou None."""
        return list(self._data.values())

    def names_tuple(self) -> tuple[str, ...]:
        """Tupla de nomes; cacheada se cache_snapshots=True."""
        if self.cache_snapshots and self._names_cache is not None:
            return self._names_cache
        tup = tuple(self._data.keys())
        if self.cache_snapshots:
            self._names_cache = tup
        return tup

    def values_tuple(self) -> tuple[Optional[ProcessModelView], ...]:
        """Tupla de valores; cacheada se cache_snapshots=True."""
        if self.cache_snapshots and self._values_cache is not None:
            return self._values_cache
        tup = tuple(self._data.values())
        if self.cache_snapshots:
            self._values_cache = tup
        return tup

    # --- Helpers específicos do domínio ---

    def add(
        self,
        name: str,
        view: Optional[ProcessModelView] = None,
        *,
        overwrite: bool = False,
    ) -> None:
        """
        Adiciona uma entrada.

        Regras:
        - Se view is None:
            - permitido apenas se o registry está vazio (primeira inserção);
            - caso contrário, será rejeitado.
        - overwrite=False evita sobrescrever uma entrada existente.
        """
        self._validate_name(name)

        if not overwrite and name in self._data:
            raise KeyError(f"'{name}' já existe no registry.")

        if view is None:
            if len(self._data) == 0 or (overwrite and name in self._data and len(self._data) == 1):
                # Permite placeholder somente como primeira entrada no registry vazio.
                # Obs.: a condição com overwrite cobre o caso de re-adicionar o mesmo nome
                # como placeholder quando o único item já é esse nome.
                self._data[name] = None
                self._invalidate_cache()
                return
            raise ValueError(
                "Adicionar um nome sem view só é permitido como a primeira inserção em um registry vazio."
            )

        # view concreta
        self._validate_view(view)
        self._data[name] = view
        self._invalidate_cache()

    def add_many(
        self,
        entries: Mapping[str, Optional[ProcessModelView]] | Iterable[tuple[str, Optional[ProcessModelView]]],
        *,
        overwrite: bool = False,
    ) -> None:
        """
        Adiciona várias entradas em lote.

        Regras para placeholders (view=None):
        - Permitido somente se o registry estiver vazio E o lote tiver exatamente 1 entrada
          E essa entrada for None.
        - Caso contrário, será rejeitado.
        """
        # Materializa para permitir validação prévia sem consumir o iterável.
        if isinstance(entries, Mapping):
            pairs = list(entries.items())
        else:
            pairs = list(entries)

        has_none = any(v is None for _, v in pairs)
        if has_none:
            # Permitimos apenas se for exatamente um par e o registry estiver vazio.
            if not (len(self._data) == 0 and len(pairs) == 1 and pairs[0][1] is None):
                raise ValueError(
                    "Adicionar nome sem view (placeholder) é permitido apenas como a primeira "
                    "inserção e isoladamente (um único item) em um registry vazio."
                )

        for name, view in pairs:
            self.add(name, view, overwrite=overwrite)

    def remove(self, name: str) -> Optional[ProcessModelView]:
        """Remove e retorna a view associada a 'name' (ou None, se placeholder)."""
        value = self._data.pop(name)
        self._invalidate_cache()
        return value

    def rename(self, old: str, new: str, *, overwrite: bool = False) -> None:
        """
        Renomeia uma entrada. Observação: ao renomear, a entrada vai para o fim
        da ordem de inserção (comportamento do dict).
        """
        if old == new:
            return
        self._validate_name(new)
        if not overwrite and new in self._data:
            raise KeyError(f"'{new}' já existe no registry.")
        value = self._data.pop(old)
        self._data[new] = value
        self._invalidate_cache()

    def get_many(
        self,
        names: Iterable[str],
        *,
        missing: Literal["error", "skip", "none"] = "error",
    ) -> list[Optional[ProcessModelView]]:
        """
        Busca várias views por nome.

        missing:
          - 'error': levanta KeyError se faltar algum nome.
          - 'skip': ignora nomes ausentes.
          - 'none': coloca None nas posições ausentes.
        """
        out: list[Optional[ProcessModelView]] = []
        if missing == "error":
            for n in names:
                out.append(self._data[n])
        elif missing == "skip":
            for n in names:
                if n in self._data:
                    out.append(self._data[n])
        elif missing == "none":
            for n in names:
                out.append(self._data.get(n))
        else:
            raise ValueError("missing deve ser 'error', 'skip' ou 'none'.")
        return out

    def compute_map(
        self,
        names: Iterable[str] | None = None,
        *,
        on_error: Literal["raise", "skip", "none"] = "raise",
    ) -> dict[str, Any]:
        """
        Materializa o resultado de compute() para várias entradas.

        on_error:
          - 'raise': propaga a exceção (inclui placeholder sem view).
          - 'skip': ignora entradas que falharem ou que sejam placeholders.
          - 'none': associa None às entradas que falharem ou que sejam placeholders.
        """
        result: dict[str, Any] = {}
        it = self._data.items() if names is None else ((n, self._data[n]) for n in names)
        for name, view in it:
            if view is None:
                if on_error == "raise":
                    raise ValueError(f"A entrada '{name}' não possui view associada (placeholder).")
                elif on_error == "skip":
                    continue
                elif on_error == "none":
                    result[name] = None
                else:
                    raise ValueError("on_error deve ser 'raise', 'skip' ou 'none'.")
                continue

            try:
                result[name] = view.compute()
            except Exception:
                if on_error == "raise":
                    raise
                elif on_error == "skip":
                    continue
                elif on_error == "none":
                    result[name] = None
                else:
                    raise ValueError("on_error deve ser 'raise', 'skip' ou 'none'.")
        return result

    def to_graphviz_map(
        self,
        names: Iterable[str] | None = None,
        *,
        on_error: Literal["raise", "skip", "none"] = "raise",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Gera visualizações via to_graphviz() para várias entradas.

        on_error:
          - 'raise': propaga exceções (inclui placeholder sem view).
          - 'skip': ignora entradas que falharem ou que sejam placeholders.
          - 'none': associa None às entradas que falharem ou que sejam placeholders.
        """
        viz: dict[str, Any] = {}
        it = self._data.items() if names is None else ((n, self._data[n]) for n in names)
        for name, view in it:
            if view is None:
                if on_error == "raise":
                    raise ValueError(f"A entrada '{name}' não possui view associada (placeholder).")
                elif on_error == "skip":
                    continue
                elif on_error == "none":
                    viz[name] = None
                else:
                    raise ValueError("on_error deve ser 'raise', 'skip' ou 'none'.")
                continue

            try:
                viz[name] = view.to_graphviz(**kwargs)
            except Exception:
                if on_error == "raise":
                    raise
                elif on_error == "skip":
                    continue
                elif on_error == "none":
                    viz[name] = None
                else:
                    raise ValueError("on_error deve ser 'raise', 'skip' ou 'none'.")
        return viz

    # --- Safety / sharing ---

    def freeze(self) -> Mapping[str, Optional[ProcessModelView]]:
        """
        Retorna uma visão de mapeamento somente leitura. Mutations no registry
        se refletem aqui, mas o mapeamento retornado é imutável para o chamador.
        """
        return MappingProxyType(self._data)

    def clear(self) -> None:
        """Remove todas as entradas."""
        self._data.clear()
        self._invalidate_cache()

    # --- Informational helpers ---

    def has_placeholder(self) -> bool:
        """Retorna True se existir pelo menos uma entrada sem view (placeholder)."""
        return any(v is None for v in self._data.values())

    # --- Internals ---

    def _invalidate_cache(self) -> None:
        self._names_cache = None
        self._values_cache = None

    @staticmethod
    def _validate_name(name: str) -> None:
        if not isinstance(name, str):
            raise TypeError("name deve ser do tipo str.")
        if not name:
            raise ValueError("name não pode ser vazio.")

    @staticmethod
    def _validate_view(view: ProcessModelView) -> None:
        if not isinstance(view, ProcessModelView):
            raise TypeError("value deve ser uma instância de ProcessModelView.")
