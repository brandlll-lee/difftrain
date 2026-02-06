#!/usr/bin/env bash
# Source this script before GPU smoke runs:
#   source ./scripts/setup_runtime_env.sh

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "This script is intended to be sourced, not executed."
  echo "Usage: source ./scripts/setup_runtime_env.sh"
  exit 1
fi

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

if [[ -z "${HF_ENDPOINT:-}" ]]; then
  if command -v curl >/dev/null 2>&1; then
    if ! curl -fsSIL --connect-timeout 8 --max-time 20 https://huggingface.co >/dev/null; then
      export HF_ENDPOINT="https://hf-mirror.com"
    fi
  fi
fi

echo "CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG}"
if [[ -n "${HF_ENDPOINT:-}" ]]; then
  echo "HF_ENDPOINT=${HF_ENDPOINT}"
else
  echo "HF_ENDPOINT=<official default>"
fi
