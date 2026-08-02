# P-MUSE Eval

P-MUSE Eval evaluates generated WAV files on three public MIDI-to-audio
benchmarks:

- `benchmark_paired`: paired audio and MIDI context.
- `benchmark_style`: an audio style prompt without prompt MIDI.
- `benchmark_mixed`: a style prompt plus paired content context.

Each benchmark has 400 generation records and 400 editing records. Every
editing record has five variants, so one complete editing submission contains
2,000 WAV files.

The P-MUSE-eval toolkit reports two metrics:

1. Instrument embedding cosine similarity.
2. Onset F1 with a 50 ms tolerance.

## Installation

```bash
git clone https://github.com/FEAfeatherTHER/P-MUSE-eval.git
cd P-MUSE-eval
pip install -r requirements.txt
```

The tested dependency versions use Python 3.10. PyTorch and the other packages
are installed from their standard package sources.

The two model repositories are external dependencies. P-MUSE Eval does not
bundle, clone, or modify them. Their paths are supplied through environment
variables.

## External model repositories

Download the Musical Instrument Embedding model to any directory:

```bash
git clone https://github.com/Alexuan/musical_instrument_embedding \
  /path/to/musical_instrument_embedding
```

P-MUSE Eval uses the YourMT3 `YPTF+Single (noPS)` model. No other
YourMT3 checkpoint is needed. Git LFS is
required.

```bash
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone \
  https://huggingface.co/spaces/mimbres/YourMT3 /path/to/YourMT3
git -C /path/to/YourMT3 lfs pull \
  --include="amt/logs/2024/ptf_all_cross_rebal5_mirst_xk2_edr005_attend_c_full_plus_b100/checkpoints/model.ckpt" \
  --exclude=""
```

## Submission layout

Use exactly this structure and these filenames:

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

`<variant>` is one of `add`, `delete`, `pitch_shift`, `velocity_scale`, or
`timing`.

Each editing WAV must contain the complete `prefix + edited target + suffix`
output. It must not contain only the edited target. For editing metrics, only
the edited target interval is scored. In the complete output, this interval
starts after the prefix and ends before the suffix.

Recommended complete submission:

```text
submission/benchmark_paired/gen/       400 WAV files
submission/benchmark_paired/edit/     2000 WAV files
submission/benchmark_style/gen/        400 WAV files
submission/benchmark_style/edit/      2000 WAV files
submission/benchmark_mixed/gen/        400 WAV files
submission/benchmark_mixed/edit/      2000 WAV files
```

## Evaluation

Set the paths, partition, and step switches:

```bash
export TESTSET=/path/to/P-MUSE-eval-testset            # Downloaded P-MUSE test set
export SUBMISSION=/path/to/submission                  # Generated WAV submission
export RESULTS=/path/to/results                        # Metric result output
export GENERATED_MIDI=/path/to/generated_midi          # Generated-audio transcription output
export INSTRUMENT_ROOT=/path/to/musical_instrument_embedding  # Instrument model repository
export YOURMT3_ROOT=/path/to/YourMT3                    # YourMT3 repository
export BENCHMARK=benchmark_paired                       # benchmark_paired, benchmark_style, or benchmark_mixed
export TASK=gen                                         # gen or edit
export DEVICE=auto                                      # auto, cpu, or cuda

export RUN_VALIDATE=1                  # 1: validate submitted WAV files; 0: skip
export RUN_INSTRUMENT=1                # 1: compute instrument similarity; 0: skip
export RUN_TRANSCRIBE=1                # 1: transcribe generated audio; 0: skip
export RUN_ONSET=1                     # 1: compute Onset F1; 0: skip
export SKIP_AUDIO_CHECK=0              # 1: validate names only; 0: also decode WAV headers
```

All four step switches default to `1`, so the complete evaluation runs with:

```bash
bash run_eval.sh
```

Set a step switch to `0` to skip it. For example, compute only instrument
similarity with previously prepared paths and files:

```bash
RUN_VALIDATE=0 RUN_TRANSCRIBE=0 RUN_ONSET=0 bash run_eval.sh
```

The script runs enabled steps in this order:

1. Validate submitted WAV names and files.
2. Compute instrument similarity.
3. Transcribe audio with YourMT3.
4. Compute Onset F1.

YourMT3 transcribes only the submitted generated audio. For generation this is
one WAV per record; for editing it is one WAV per edit variant. Transcriptions
and their cache records are stored under
`<GENERATED_MIDI>/<benchmark>/<task>/`.

Each run evaluates one `$BENCHMARK/$TASK`. Change those two variables for another
partition. Instrument similarity is independent. Onset F1 requires the MIDI and
cache records produced by the transcription step; only disable transcription
when those files already exist and match the current submitted WAV files.

## Metric definitions

### Instrument similarity

All prompt audio provided for a sample is concatenated and converted into one
512-dimensional embedding. The generated target audio is converted into another
embedding, and their cosine similarity is the sample's instrument score.
Instrument embeddings are recalculated for every sample on every run and are
not stored by the toolkit.

For generation, the submitted WAV is the generated target audio. For editing,
the toolkit extracts only the edited target interval from the submitted
`prefix + edited target + suffix` WAV; the prefix and suffix remain prompt audio.
If a separate style prompt is provided, it is included with the other prompt
audio.

### Onset F1

Generation compares the full generated transcription against the benchmark
target MIDI. For `benchmark_paired` and `benchmark_mixed`, the target interval
is extracted from `midi_path` and shifted to start at zero. For
`benchmark_style`, `target_midi_path` is used directly.

Editing compares the transcription inside the generated edited target interval
against the corresponding edited target MIDI. Generated onsets in this interval
are shifted so the interval starts at zero. Editing does not transcribe or score
`ref_audio/edit`.

The metric calls `mir_eval.transcription.onset_precision_recall_f1` with a
50 ms tolerance.

#### P-MUSE paper protocol

The P-MUSE paper used a stricter generation protocol: both the generated audio
and the ground-truth target audio were transcribed with YourMT3, then Onset F1
was computed between the two transcriptions. This reduces mismatch between a
performed audio recording and its source MIDI. The toolkit default uses the
benchmark target MIDI for a simpler and more general workflow, so its generation
Onset F1 values are not directly comparable with those reported in the paper.
