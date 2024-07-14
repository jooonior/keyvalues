import functools
import io
import re
from typing import Any

import keyvalues
from keyvalues import Token


def trace(decorator):
    prefix = decorator.__name__

    @functools.wraps(decorator)
    def trace_decorator(parser):
        parser = decorator(parser)

        @functools.wraps(parser)
        def decorated(tokens, depth):
            for token in parser(tokens, depth):
                data = repr(token)

                if token.role is not None:
                    data = f"{token.role.name}({data})"

                print(f"{prefix}: {data}")  # noqa: T201
                yield token

        return decorated

    return trace_decorator


def pipeline(parser, *decorators):
    def printer(prefix, tokens):
        for token in tokens:
            print(f"{prefix}: {token!r}")  # noqa: T201
            yield token

    @functools.wraps(parser)
    def printing_parser(tokens, depth):
        tokens = printer("lexer", tokens)
        tokens = parser(tokens, depth)
        tokens = printer("parser", tokens)
        return tokens  # noqa: RET504

    return keyvalues.pipeline(printing_parser, *map(trace, decorators))


def lexer(string):
    return keyvalues.lexer(io.StringIO(string))


def writer(tokens):
    file = io.StringIO()
    keyvalues.writer(tokens, file)
    return file.getvalue()


def test_loader():
    tokens = lexer("k1 v1 [c1] k2 { k3 v3 k4 [c4] v4 k5 {} } k6 [c6] {}")
    tokens = pipeline(keyvalues.parser)(tokens, 0)
    # fmt: off
    assert keyvalues.loader(tokens, list[Any]) == [
        ((Token("k1"), Token("[c1]")), Token("v1")),
        ((Token("k2"), None), [
            ((Token("k3"), None), Token("v3")),
            ((Token("k4"), Token("[c4]")), Token("v4")),
            ((Token("k5"), None), []),
        ]),
        ((Token("k6"), Token("[c6]")), []),
    ]
    # fmt: on


def test_macros():
    parser = pipeline(
        keyvalues.parser,
        keyvalues.parse_macros("#1", "#2"),
    )

    tokens = lexer("#0 v #1 [] k { #1 [] v } #2 {")
    tokens = parser(tokens, 0)

    # fmt: off
    assert keyvalues.loader(tokens, list[Any]) == [
        ((Token("#0"), None), Token("v")),
        ((Token("#1"), None), Token("[]")),
        ((Token("k"), None), [
            ((Token("#1"), Token("[]")), Token("v")),
        ]),
        ((Token("#2"), None), Token("{")),
    ]
    # fmt: on


def test_errors_and_macros():
    parser = pipeline(
        keyvalues.parser,
        keyvalues.report_errors(),
        keyvalues.parse_macros("#base", "#include"),
    )

    tokens = lexer("#base } k1 { k2 v2 }")
    tokens = parser(tokens, 0)

    # fmt: off
    assert keyvalues.loader(tokens, list[Any]) == [
        ((Token("#base"), None), Token("}")),
        ((Token("k1"), None), [
            ((Token("k2"), None), Token("v2")),
        ]),
    ]
    # fmt: on


def deindent(string):
    string = string.removeprefix("\n")
    indent = min(
        re.match(r"^[^\S\n]*", line).end()  # type: ignore[union-attr]
        for line in string.split("\n")
        if line
    )
    return re.sub(rf"^[^\S\n]{{{indent}}}", "", string, flags=re.MULTILINE)


def test_formatter_default():
    tokens = lexer('1 { "2" [] { 3 c [] } } 4 [] {}')
    parser = pipeline(
        keyvalues.parser,
        keyvalues.formatter(indent="    "),
    )
    tokens = parser(tokens, 0)
    assert writer(tokens) == deindent(
        """
        1
        {
            "2" []
            {
                3 c []
            }
        }
        4 []
        {
        }
        """
    )


def test_formatter_compact():
    tokens = lexer('1 { "2" [] { 3 c [] } } 4 [] {}')
    parser = pipeline(
        keyvalues.parser,
        keyvalues.formatter(
            indent="    ",
            newline_before_section=False,
            collapse_empty_sections=True,
        ),
    )
    tokens = parser(tokens, 0)
    assert writer(tokens) == deindent(
        """
        1 {
            "2" [] {
                3 c []
            }
        }
        4 [] {}
        """
    )


def test_formatter_blanks():
    tokens = lexer("\n\n\n1\n\n\n2\n\n\n3\n\n\n4\n\n\n")
    parser = pipeline(
        keyvalues.parser,
        keyvalues.formatter(
            indent="    ",
            max_consecutive_blank_lines=1,
        ),
    )
    tokens = parser(tokens, 0)
    assert writer(tokens) == deindent(
        """
        1 2

        3 4
        """
    )


def test_formatter_comments():
    tokens = lexer("a //1\n { b  //2 \n[] c\n\n//3\n }//=")
    parser = pipeline(
        keyvalues.parser,
        keyvalues.formatter(
            indent="    ",
            max_consecutive_blank_lines=0,
        ),
    )
    tokens = parser(tokens, 0)
    assert writer(tokens) == deindent(
        """
        a  // 1
        {
            b  // 2
            [] c
            // 3
        }  //=
        """
    )
