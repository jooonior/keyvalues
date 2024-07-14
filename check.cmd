#!/usr/bin/env -S 2>/dev/null=2>NUL sh -x

mypy --no-error-summary . || exit 1
ruff check --quiet . || exit 1
ruff format --diff --quiet . || exit 1
