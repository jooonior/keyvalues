from __future__ import annotations

import collections
import enum
import itertools
import re
from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from typing import (
        Any,
        ParamSpec,
        Self,
    )


if TYPE_CHECKING:
    Params = ParamSpec("Params")
    Ret = TypeVar("Ret")


def copy_signature_from(
    _origin: Callable[Params, Ret],
) -> Callable[[Callable[..., Any]], Callable[Params, Ret]]:
    def decorator(target: Callable[..., Any]) -> Callable[Params, Ret]:
        return target

    return decorator


T = TypeVar("T")


class CaseInsensitiveDict(Generic[T], collections.UserDict[str, T]):
    def __getitem__(self, key: str) -> T:
        return super().__getitem__(key.lower())

    def __setitem__(self, key: str, value: T) -> None:
        return super().__setitem__(key.lower(), value)

    def __delitem__(self, key: str) -> None:
        return super().__delitem__(key.lower())

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str):
            key = key.lower()
        return super().__contains__(key)


N = TypeVar("N", int, float, Decimal)


@dataclass
class xrange(Generic[N]):  # noqa: N801
    start: N
    end: N
    step: N

    def __post_init__(self) -> None:
        if self.step == 0:
            errmsg = "step is zero"
            raise ValueError(errmsg)

        if self.step < 0 if self.start <= self.end else self.step > 0:
            errmsg = "range is infinite"
            raise ValueError(errmsg)

    def __iter__(self) -> Iterator[N]:
        n = self.start

        if self.step > 0:
            while n < self.end:
                yield n
                n += self.step
        else:
            while n > self.end:
                yield n
                n += self.step


class Peekable(Generic[T], Iterator[T]):
    def __init__(self, iterable: Iterable[T]):
        self._it = iter(iterable)
        self._next = collections.deque[T]()

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> T:
        return self._next.popleft() if self._next else next(self._it)

    def __getitem__(self, index: int) -> T:
        if index < 0:
            errmsg = "negative index"
            raise ValueError(errmsg)

        self._load(index + 1)
        return self._next[index]

    def get(self, index: int, default: T | None = None) -> T | None:
        try:
            return self[index]
        except IndexError:
            return default

    def skip(self, count: int) -> None:
        next(itertools.islice(self, count, count), None)

    def _load(self, count: int) -> None:
        loaded = len(self._next)
        if count > loaded:
            self._next.extend(itertools.islice(self._it, count - loaded))

    def __bool__(self) -> bool:
        self._load(1)
        return bool(self._next)


class AutoIntEnum(enum.IntEnum):
    def __new__(cls, value: int | None = None) -> Self:
        if value is None:
            value = len(cls)
        obj = int.__new__(cls, value)
        obj._value_ = value
        return obj


if TYPE_CHECKING:
    # Make type checker think that `RegexEnum` has attributes of `re.Pattern`.

    class RegexEnumMeta(enum.EnumType, re.Pattern[str]):  # type: ignore[misc]
        pass

else:
    # When in reality, those attributes are handled by `__getattr__`.

    class RegexEnumMeta(enum.EnumType):
        def __getattr__(cls, name: str) -> Any:
            if cls is RegexEnum:
                return getattr(super(), name)

            attr = getattr(cls._re, name)
            setattr(cls, name, attr)
            return attr


class RegexEnum(enum.IntEnum, metaclass=RegexEnumMeta):
    _ignore_ = "pattern"

    pattern: str

    if TYPE_CHECKING:
        # Type checker thinks that `__new__` is used to lookup members.
        def __new__(cls, value: int) -> Self: ...

    else:

        def __new__(cls, value: str | None = None) -> Self:
            index = len(cls) + 1

            if value is None:
                index *= -1
                value = ""

            obj = int.__new__(cls, index)
            obj._value_ = index
            obj.pattern = value

            return obj

    def __init_subclass__(cls) -> None:
        patterns = []

        for member in cls:
            # Convert outermost capturing group into named capturing group.
            open_named_group = f"(?P<{member.name}>"
            regex, n = re.subn(
                r"(?<!\\)\((?!\?)", open_named_group, member.pattern, count=1
            )
            if n == 0:
                # No existing capture group, wrap whole regex in one.
                patterns.append(f"{open_named_group}{regex})")
            else:
                # Wrap regex in non-capturing group to ensure correct
                # interpretation when multiple regexes are joined by "|".
                patterns.append(f"(?:{regex})")

        cls._re = re.compile("|".join(patterns))  # type: ignore[attr-defined]
