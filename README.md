# P-MUSE-eval v3

P-MUSE-eval evaluates generated WAV files on `benchmark_paired`,
`benchmark_style`, and `benchmark_mixed`. Each benchmark contains generation
and editing tasks for `piano`, `guitar`, `bass`, and `drum`.

It reports two metrics:

1. Instrument embedding cosine similarity between generated target audio and
   prompt audio.
2. Onset F1. Onsets must be within 50 ms and have the exact MIDI pitch by
   default.

Generated audio is transcribed with MuScriptor `large`. Each record's `family`,
`dataset_name`, and `instrument_name` in `metadata.jsonl` select exactly one
MuScriptor instrument group, preventing a single-instrument recording from
being decoded into duplicate acoustic/electric tracks.
Tempo detection is disabled to preserve predicted note times. Transcriptions
cached with tempo detection enabled are regenerated on the next transcription run.

## Installation

Python 3.10 and a CUDA-capable PyTorch installation are recommended.

```bash
conda create -n P-MUSE-eval python=3.10 -y
conda activate P-MUSE-eval
pip install -r requirements.txt
```

To update the existing environment used in this repository:

```bash
PIP_CONFIG_FILE=/dev/null \
/root/miniconda3/envs/P-MUSE-eval/bin/python -m pip install \
  --no-user --upgrade \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  -r /data/250010171/code/_midi2music/P-MUSE-eval-v3/requirements.txt
```

## Model files

Download the instrument embedding repository:

```bash
git clone https://github.com/Alexuan/musical_instrument_embedding \
  /path/to/musical_instrument_embedding
```

MuScriptor `large` is the most accurate published variant. Its weights use the
CC BY-NC 4.0 license and require accepting the access conditions on
[Hugging Face](https://huggingface.co/MuScriptor/muscriptor-large). Put both
downloaded files in the same server directory:

```text
/mnt/data/jingchong/models/muscriptor-large/
  config.json
  model.safetensors
```

For example, upload the local directory with `rsync`:

```bash
rsync -avP /local/path/muscriptor-large/ \
  user@server:/mnt/data/jingchong/models/muscriptor-large/
```

## Submission layout

```text
submission/
  benchmark_paired/
    gen/<record_id>.wav
    edit/<record_id>__<variant>.wav
  benchmark_style/
    gen/<record_id>.wav
    edit/<record_id>__<variant>.wav
  benchmark_mixed/
    gen/<record_id>.wav
    edit/<record_id>__<variant>.wav
```

Editing variants are `add`, `delete`, `pitch_shift`, `velocity_scale`, and
`timing`. Editing WAV files contain the complete prefix, edited target, and
suffix; only the target interval is scored.

## Evaluation

```bash
export TESTSET=/path/to/P-MUSE-eval-testset
export SUBMISSION=/path/to/submission
export RESULTS=/path/to/results
export GENERATED_MIDI=/path/to/generated_midi
export INSTRUMENT_ROOT=/path/to/musical_instrument_embedding
export DEVICE=auto
export MUSCRIPTOR_MODEL=/mnt/data/jingchong/models/muscriptor-large/model.safetensors

export RUN_VALIDATE=1
export RUN_INSTRUMENT=1
export RUN_TRANSCRIBE=1
export RUN_ONSET=1
export ONSET_MATCH_PITCH=1
```

Evaluate one benchmark/task pair:

```bash
BENCHMARK=benchmark_paired TASK=gen bash run_eval.sh
```

Evaluate all six pairs:

```bash
for benchmark in benchmark_paired benchmark_style benchmark_mixed; do
  for task in gen edit; do
    BENCHMARK="$benchmark" TASK="$task" bash run_eval.sh
  done
done
```

The four steps are validation, instrument similarity, MuScriptor transcription,
and Onset F1. Transcribed MIDI and cache records are stored under
`<GENERATED_MIDI>/<benchmark>/<task>/`. Results are written to the `instrument/`
and `onset/` directories under `<RESULTS>/<benchmark>/<task>/`. Existing offset
result directories from older evaluator versions are not read or updated.
