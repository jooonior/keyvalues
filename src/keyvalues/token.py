from __future__ import annotations

import dataclasses
import re
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar

if TYPE_CHECKING:
    from collections.abc import (
        Iterator,
    )
    from typing import (
        Self,
        Unpack,
    )


# The type hint assert below doesn't work with the new type parameter syntax.
Tag = TypeVar("Tag", bound=int)
Meta = TypeVar("Meta", bound=Mapping[str, Any])


class TokenKwArgs(Generic[Tag, Meta], TypedDict, total=False):
    tag: Tag
    meta: Meta
    filename: str | None
    line: str
    lineno: int
    start: int
    data_start: int
    end: int


class TokenFields(Generic[Tag, Meta], TokenKwArgs[Tag, Meta], total=False):
    data: str


@dataclass(slots=True)
class Token(Generic[Tag, Meta]):
    data: str
    _: dataclasses.KW_ONLY
    tag: Tag
    meta: Meta = dataclasses.field(default_factory=dict)  # type: ignore[assignment]
    filename: str | None = None
    line: str = None  # type: ignore[assignment]
    lineno: int = 0
    start: int = 0
    data_start: int = 0
    end: int = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.line is None:
            self.line = self.data  # type: ignore[unreachable]

        # When only `start` is specified.
        self.data_start = max(self.start, self.data_start)

        if self.end is None:
            self.end = self.data_start + len(self.data)  # type: ignore[unreachable]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Token):
            return False

        return self.data == other.data and self.tag == other.tag

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return repr(self.data)

    def clone(self, **kwargs: Unpack[TokenFields[Tag, Meta]]) -> Self:
        # Copy our metadata and merge it with what was passed.
        kwargs["meta"] = {**self.meta, **kwargs.get("meta", {})}  # type: ignore[typeddict-item]
        return dataclasses.replace(self, **kwargs)

    def __getitem__(self, index: int | slice) -> Self:
        if isinstance(index, slice):
            start, end, step = index.indices(len(self.data))
            if step != 1:
                errmsg = "step not supported in slice"
                raise NotImplementedError(errmsg)
        else:
            start = index if index >= 0 else len(self.data) + index
            end = start + 1

        start += self.data_start
        end += self.data_start

        return self.clone(
            data=self.data[index],
            start=start,
            data_start=start,
            end=end,
        )

    def split(self, delims: str) -> Iterator[Self]:
        for match in re.finditer(rf"[^{delims}]+", self.data):
            yield self[match.start() : match.end()]

    @classmethod
    def from_match(
        cls,
        match: re.Match[str],
        **kwargs: Unpack[TokenFields[Tag, Meta]],
    ) -> Self:
        lastindex = match.lastindex
        if lastindex is None:
            errmsg = "match does not capture any groups"
            raise ValueError(errmsg)

        kwargs.setdefault("data", match[lastindex])
        kwargs.setdefault("tag", lastindex)  # type: ignore[typeddict-item]
        kwargs.setdefault("line", match.string)
        kwargs.setdefault("start", match.start())
        kwargs.setdefault("data_start", match.start(lastindex))
        kwargs.setdefault("end", match.end())

        return cls(**kwargs)

    @classmethod
    def from_other(
        cls,
        other: Token[Any, Any],
        *,
        tag: Tag,
        meta: Meta | None = None,
    ) -> Self:
        if meta is None:
            meta = typing.cast(Meta, {})

        fields = {field.name for field in dataclasses.fields(cls)}
        fields.intersection_update(
            field.name for field in dataclasses.fields(type(other))
        )
        fields.difference_update(["tag", "meta"])

        return cls(
            **{field: getattr(other, field) for field in fields},
            tag=tag,
            meta=meta,
        )


assert typing.get_type_hints(Token) == typing.get_type_hints(TokenFields) | {
    "_": dataclasses.KW_ONLY
}


if TYPE_CHECKING:
    AnyToken = Token[Any, Any]


class TokenError(SyntaxError):
    def __init__(self, message: str, token: Token[Any, Any]):
        super().__init__(message)
        self.token = token

    # Type checker complains because properties have no setters.

    @property
    def filename(self) -> str | None:  # type: ignore[override]
        return self.token.filename

    @property
    def lineno(self) -> int:  # type: ignore[override]
        return self.token.lineno

    @property
    def offset(self) -> int:  # type: ignore[override]
        return self.token.start + 1

    @property
    def text(self) -> str:  # type: ignore[override]
        return self.token.line

    @property
    def end_lineno(self) -> int:  # type: ignore[override]
        return self.token.lineno

    @property
    def end_offset(self) -> int:  # type: ignore[override]
        return self.token.end + 1
