from __future__ import annotations

import argparse
import collections
import re
import sys
import traceback
from typing import TYPE_CHECKING

from . import dbg, fmt, parse, pp
from .token import TokenError

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

    return parser


def build_format_pipeline_from_arguments(
    args: argparse.Namespace,
) -> Iterator[parse.ParserDecorator]:
    yield fmt.normalize(
        indent=args.indent,
    )


def main() -> int:
    """Run the `keyvalues` CLI. Returns the intended exit code."""
    args = make_argument_parser().parse_args()

    pipeline = parse.Pipeline()
    pipeline.add(parse.parse_macros("#base", "#include"))

    try:
        match args.command:
            case "check":
                for file in args.files:
                    tokens = pipeline.parse(file)
                    # Consume all tokens.
                    collections.deque(tokens, maxlen=0)

            case "format":
                pipeline.add(
                    *build_format_pipeline_from_arguments(args),
                )
                tokens = pipeline.parse(args.file)
                fmt.writer(tokens, sys.stdout)

            case "expand":
                pipeline.add(
                    pp.preprocess,
                    *build_format_pipeline_from_arguments(args),
                )
                tokens = pipeline.parse(args.file)
                fmt.writer(tokens, sys.stdout)

    except TokenError as e:
        traceback.print_exception(e, limit=(None if dbg.DEBUG else 0))
        return 1

    else:
        return 0
