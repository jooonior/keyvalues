from __future__ import annotations

import argparse
import collections
import re
import sys
import traceback
from typing import TYPE_CHECKING

from . import fmt, parse, proc

if TYPE_CHECKING:
    from collections.abc import Iterator


class SubcommandHelpFormatter(argparse.HelpFormatter):
    """Custom help message formatter."""

    def _format_action(self, action: argparse.Action) -> str:
        parts = super()._format_action(action)
        if action.nargs == argparse.PARSER:
            # Remove metavar from subparser list and decrease its indentation.
            parts = re.sub(r"^.*\n  |(\n)  ", r"\1", parts)
        return parts


def make_argument_parser() -> argparse.ArgumentParser:
    prog = __package__.split(".")[-1]  # the package name

    parser = argparse.ArgumentParser(
        prog=prog,
        formatter_class=SubcommandHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        title="commands",
        metavar="command",
    )

    check_subparser = subparsers.add_parser(
        "check",
        help="validate syntax of input file(s)",
        description="Check the syntax of input file(s) and report any errors.",
    )
    check_subparser.add_argument(
        "files",
        nargs="+",
        type=argparse.FileType("r"),
        default=[sys.stdin],
        help="input file",
        metavar="file",
    )
    check_subparser.add_argument(
        "-d",
        "--directives",
        action="store_true",
        help=f'parse preprocessor directives (see "{prog} expand -h")',
    )

    file_parser = argparse.ArgumentParser(add_help=False)
    file_parser.add_argument(
        "file",
        type=argparse.FileType("r"),
        help="input file",
    )

    format_parser = make_format_argument_parser()

    subparsers.add_parser(
        "format",
        help="format input file",
        description=(
            "Format the input file and print the formatted output to stdout."
        ),
        parents=[file_parser, format_parser],
    )

    subparsers.add_parser(
        "expand",
        help="expand directives and expressions in input file",
        description=(
            "Expand directives and expression in the input file and print "
            "the expanded output to stdout. "
        ).strip(),
        parents=[file_parser, format_parser],
    )

    return parser


def make_format_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_argument_group("output format options")

    # Not a lambda because error messages shows the function name.
    def indentation(string: str) -> str:
        return string if string.isspace() else " " * int(string)

    group.add_argument(
        "-i",
        "--indent",
        type=indentation,
        default="\t",
        help=(
            "indentation size in spaces per level "
            "(default: use single TAB instead) "
        ).strip(),
    )

    group.add_argument(
        "-q",
        "--quote",
        type=re.compile,
        help="regex matching tokens to quote (default: preserve quotes)",
    )

    group.add_argument(
        "-b",
        "--maxblanks",
        type=int,
        default=1,
        help="maximum number of consecutive blank lines (default: 1)",
    )

    group.add_argument(
        "-c",
        "--compact",
        action="store_true",
        help="put { on the same line as section key",
    )

    group.add_argument(
        "-m",
        "--minify",
        action="store_true",
        help="minify output (overrides other format options)",
    )

    return parser


def build_format_pipeline_from_arguments(
    args: argparse.Namespace,
) -> Iterator[parse.ParserDecorator]:
    if args.minify:
        yield fmt.autoquote()
        yield fmt.minify
        return

    if args.quote:
        yield fmt.autoquote(args.quote.pattern)

    yield fmt.normalize_whitespace(
        indent=args.indent,
        newline_before_section=not args.compact,
        collapse_empty_sections=args.compact,
        max_consecutive_blank_lines=args.maxblanks,
    )


def main() -> int:
    """Run the `keyvalues` CLI. Returns the indended exit code."""
    args = make_argument_parser().parse_args()

    pipeline = parse.Pipeline()
    standard_behavior = [
        parse.report_errors(treat_empty_root_key_as_error=False),
        parse.parse_macros("#base", "#include"),
    ]

    try:
        match args.command:
            case "check":
                for file in args.files:
                    pipeline.add(*standard_behavior)
                    tokens = pipeline.parse(file)
                    # Consume all tokens.
                    collections.deque(tokens, maxlen=0)

            case "format":
                pipeline.add(
                    *standard_behavior,
                    *build_format_pipeline_from_arguments(args),
                )
                tokens = pipeline.parse(args.file)
                fmt.writer(tokens, sys.stdout)

            case "expand":
                pipeline.add(
                    proc.expand,
                    *standard_behavior,
                    proc.merge,
                    *build_format_pipeline_from_arguments(args),
                )
                tokens = pipeline.parse(args.file)
                fmt.writer(tokens, sys.stdout)

    except parse.TokenError as e:
        traceback.print_exception(e, limit=0)
        return 1

    else:
        return 0
