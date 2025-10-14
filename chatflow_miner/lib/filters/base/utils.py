import pandas as pd

from .exceptions import FilterError


def _ensure_bool_series(mask: pd.Series, df: pd.DataFrame) -> None:
    """Garante que a série ``mask`` seja booleana e alinhada ao ``df``.

    Lança ``FilterError`` com mensagens claras caso algo esteja desalinhado ou
    com tipo inadequado — útil para depuração e para manter o contrato da API.
    """
    if not isinstance(mask, pd.Series):
        raise FilterError("Máscara deve ser uma pandas.Series")
    if mask.dtype != bool:
        # Aceita categorias/objetos "truthy"? Não; força booleana para evitar surpresas.
        try:
            mask = mask.astype(bool)
        except Exception as exc:  # pragma: no cover
            raise FilterError("Máscara não booleana e conversão falhou") from exc
    if not mask.index.equals(df.index):
        raise FilterError("Índice da máscara não corresponde ao índice do DataFrame")
