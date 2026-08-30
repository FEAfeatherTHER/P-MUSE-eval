#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

RUN_VALIDATE="${RUN_VALIDATE:-1}"
RUN_INSTRUMENT="${RUN_INSTRUMENT:-1}"
RUN_TRANSCRIBE="${RUN_TRANSCRIBE:-1}"
RUN_ONSET="${RUN_ONSET:-1}"
ONSET_MATCH_PITCH="${ONSET_MATCH_PITCH:-1}"
SKIP_AUDIO_CHECK="${SKIP_AUDIO_CHECK:-0}"
DEVICE="${DEVICE:-auto}"
MUSCRIPTOR_MODEL="${MUSCRIPTOR_MODEL:-large}"

require_binary_switch() {
  local name="$1"
  local value="${!name}"
  if [[ "$value" != "0" && "$value" != "1" ]]; then
    echo "$name must be 0 or 1, got: $value" >&2
    exit 2
  fi
}

require_variable() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Required environment variable is not set: $name" >&2
    exit 2
  fi
}

for name in \
  RUN_VALIDATE RUN_INSTRUMENT RUN_TRANSCRIBE RUN_ONSET \
  ONSET_MATCH_PITCH SKIP_AUDIO_CHECK; do
  require_binary_switch "$name"
done

if [[ "$RUN_VALIDATE" == "1" || "$RUN_INSTRUMENT" == "1" || \
      "$RUN_TRANSCRIBE" == "1" || "$RUN_ONSET" == "1" ]]; then
  require_variable BENCHMARK
  require_variable TASK
  case "$BENCHMARK" in
    benchmark_paired|benchmark_style|benchmark_mixed) ;;
    *)
      echo "BENCHMARK must be benchmark_paired, benchmark_style, or benchmark_mixed" >&2
      exit 2
      ;;
  esac
  case "$TASK" in
    gen|edit) ;;
    *)
      echo "TASK must be gen or edit" >&2
      exit 2
      ;;
  esac
fi

if [[ "$RUN_VALIDATE" == "1" ]]; then
  require_variable TESTSET
  require_variable SUBMISSION
  args=(
    --testset-root "$TESTSET"
    --submission-root "$SUBMISSION"
    --benchmark "$BENCHMARK"
    --task "$TASK"
  )
  if [[ "$SKIP_AUDIO_CHECK" == "1" ]]; then
    args+=(--skip-audio-check)
  fi
  echo "[1/4] Validate $BENCHMARK/$TASK"
  "$PYTHON_BIN" -m pmuse_eval.validate "${args[@]}"
fi

if [[ "$RUN_INSTRUMENT" == "1" ]]; then
  for name in TESTSET SUBMISSION RESULTS INSTRUMENT_ROOT; do
    require_variable "$name"
  done
  echo "[2/4] Instrument similarity $BENCHMARK/$TASK"
  "$PYTHON_BIN" -m pmuse_eval.instrument \
    --testset-root "$TESTSET" \
    --submission-root "$SUBMISSION" \
    --benchmark "$BENCHMARK" \
    --task "$TASK" \
    --results-dir "$RESULTS" \
    --instrument-embedding-root "$INSTRUMENT_ROOT" \
    --device "$DEVICE"
fi

if [[ "$RUN_TRANSCRIBE" == "1" ]]; then
  for name in TESTSET SUBMISSION GENERATED_MIDI; do
    require_variable "$name"
  done
  echo "[3/4] Transcribe audio $BENCHMARK/$TASK"
  "$PYTHON_BIN" -m pmuse_eval.transcription \
    --testset-root "$TESTSET" \
    --submission-root "$SUBMISSION" \
    --benchmark "$BENCHMARK" \
    --task "$TASK" \
    --generated-midi-dir "$GENERATED_MIDI" \
    --model "$MUSCRIPTOR_MODEL" \
    --device "$DEVICE"
fi

if [[ "$RUN_ONSET" == "1" ]]; then
  for name in TESTSET GENERATED_MIDI RESULTS; do
    require_variable "$name"
  done
  echo "[4/4] Onset F1 $BENCHMARK/$TASK"
  args=(
    --testset-root "$TESTSET"
    --benchmark "$BENCHMARK"
    --task "$TASK"
    --generated-midi-dir "$GENERATED_MIDI"
    --results-dir "$RESULTS"
  )
  if [[ "$ONSET_MATCH_PITCH" == "0" ]]; then
    args+=(--no-match-pitch)
  fi
  "$PYTHON_BIN" -m pmuse_eval.onset "${args[@]}"
fi
