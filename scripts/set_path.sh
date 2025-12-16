#!/usr/bin/env sh
set -e

# Resolve repository root and make src importable
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo_root=$(CDPATH= cd -- "${script_dir}/.." && pwd -P)
if [ -z "${PYTHONPATH:-}" ]; then
  export PYTHONPATH="${repo_root}"
else
  export PYTHONPATH="${repo_root}:${PYTHONPATH}"
fi