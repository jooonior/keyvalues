from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from typing import TextIO

from . import dbg
from .parse import (
    ParsedToken,
    ParsedTokenRole,
    ParsedTokenTag,
    parser_decorator,
)

if TYPE_CHECKING:
    from .parse import ParserDecorator, ParserIO


def normalize(
    indent: int | str = "\t",
) -> ParserDecorator:
    if isinstance(indent, int):
        indent = " " * indent

    @parser_decorator
    def normalize(
        tokens: ParserIO,
        depth: int,
    ) -> Iterator[ParsedToken]:
        def nextline() -> Iterator[ParsedToken]:
            # First call does nothing, only redefines self.
            nonlocal nextline

            def nextline() -> Iterator[ParsedToken]:
                yield ParsedToken("\n", tag=ParsedTokenTag.SPACE)
                yield ParsedToken(indent * depth, tag=ParsedTokenTag.SPACE)

            return iter([])

        for token in tokens.output:
            match token.meta.get("role"):
                case None:
                    if token.tag == ParsedTokenTag.EOF:
                        yield token

                case ParsedTokenRole.KEY | ParsedTokenRole.MACRO:
                    yield from nextline()
                    yield token.clone(tag=ParsedTokenTag.QUOTED)

                case ParsedTokenRole.VALUE:
                    yield ParsedToken(" ", tag=ParsedTokenTag.SPACE)
                    yield token.clone(tag=ParsedTokenTag.QUOTED)

                case ParsedTokenRole.CONDITION:
                    yield ParsedToken(" ", tag=ParsedTokenTag.SPACE)
                    yield token

                case ParsedTokenRole.OPEN:
                    yield from nextline()
                    yield token
                    depth += 1

                case ParsedTokenRole.CLOSE:
                    depth -= 1
                    yield from nextline()
                    yield token

                case _ as unreachable:
                    assert_never(unreachable)

        yield ParsedToken("\n", tag=ParsedTokenTag.SPACE)

    return normalize


@dbg.consumer
def writer(tokens: Iterable[ParsedToken], file: TextIO) -> None:
    """Write `tokens` to `file`."""
    for token in tokens:
        match token.tag:
            case ParsedTokenTag.PLAIN | ParsedTokenTag.SPACE:
                data = token.data
            case ParsedTokenTag.QUOTED:
                data = f'"{token.data}"'
            case ParsedTokenTag.COMMENT:
                data = f"//{token.data}"
            case ParsedTokenTag.EOF:
                continue
            case _ as unreachable:
                assert_never(unreachable)

        file.write(data)
