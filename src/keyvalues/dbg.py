from __future__ import annotations

import enum
import functools
import os
import sys
from typing import TYPE_CHECKING, assert_never

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterable,
        Iterator,
    )
    from typing import (
        Any,
        Concatenate,
        ParamSpec,
        Protocol,
        TypeVar,
    )

from . import utils

if TYPE_CHECKING:
    from .parse import ParsedToken


class DebugLevel(enum.Enum):
    QUIET = 0
    DEBUG = 1
    TRACE = 2

    def __bool__(self) -> bool:
        return bool(self.value)

    @classmethod
    def _missing_(cls, value: object) -> DebugLevel:
        if isinstance(value, int):
            return cls.QUIET if value < 0 else cls.TRACE

        try:
            value = int(value)  # type: ignore[call-overload]
        except ValueError:
            return cls.QUIET
        else:
            return cls(value)


DEBUG = DebugLevel(os.environ.get("DEBUG", ""))


@utils.copy_signature_from(print)
def log(*args: object, **kwargs: Any) -> None:
    kwargs.setdefault("file", sys.stderr)
    print(*args, **kwargs)  # noqa: T201


@utils.copy_signature_from(print)
def log_nop(*args: object, **kwargs: Any) -> None:
    pass


if TYPE_CHECKING:

    class ConsumedFn(Protocol):
        def __call__(
            self,
            tokens: Iterator[ParsedToken],
            consumer: Callable[..., Any],
        ) -> Iterator[ParsedToken]: ...


def consumed_check(
    tokens: Iterator[ParsedToken],
    consumer: Callable[..., Any],  # noqa: ARG001
) -> Iterator[ParsedToken]:
    # Import moved here to break circular dependencies.
    from .parse import ParsedTokenTag

    eof = False
    for token in tokens:
        assert not eof, "tokens after EOF"
        eof = token.tag == ParsedTokenTag.EOF
        yield token


def consumed_trace(
    tokens: Iterator[ParsedToken],
    consumer: Callable[..., Any],
) -> Iterator[ParsedToken]:
    name = getattr(consumer, "__name__", None)
    if name is None:
        name = repr(consumer)

    for token in consumed_check(tokens, consumer):
        print(f"{name:16}{token!r}", file=sys.stderr)  # noqa: T201
        yield token


def consumed_nop(
    tokens: Iterator[ParsedToken],
    consumer: Callable[..., Any],  # noqa: ARG001
) -> Iterator[ParsedToken]:
    return iter(tokens)


consumed: ConsumedFn

if not TYPE_CHECKING:
    match DEBUG:
        case DebugLevel.QUIET:
            consumed = consumed_nop
            log = log_nop
        case DebugLevel.DEBUG:
            consumed = consumed_check
        case DebugLevel.TRACE:
            consumed = consumed_trace
        case _ as unreachable:
            assert_never(unreachable)


if TYPE_CHECKING:
    Ret = TypeVar("Ret")
    Params = ParamSpec("Params")


def consumer(
    fn: Callable[Concatenate[Iterable[ParsedToken], Params], Ret],
) -> Callable[Concatenate[Iterable[ParsedToken], Params], Ret]:
    @functools.wraps(fn)
    def decorated(
        tokens: Iterable[ParsedToken],
        /,
        *args: Params.args,
        **kwargs: Params.kwargs,
    ) -> Ret:
        tokens = consumed(iter(tokens), fn)
        return fn(tokens, *args, **kwargs)

    return decorated
