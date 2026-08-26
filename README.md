# HEAR

**Run 23 speech language models on your own audio task, through one interface.**

[![EMNLP 2026](https://img.shields.io/badge/EMNLP_2026-Main_Conference-b31b1b)](https://attributetoreason.github.io/AttributeToReason/)
[![arXiv](https://img.shields.io/badge/arXiv-2608.29120-b31b1b)](https://arxiv.org/abs/2608.29120)
[![Project page](https://img.shields.io/badge/🌐_Project_page-1f6dc8)](https://attributetoreason.github.io/AttributeToReason/)
[![Dataset](https://img.shields.io/badge/🤗_HEAR-ffcc4d)](https://huggingface.co/datasets/PleasedPenguin/HEAR)
[![Model](https://img.shields.io/badge/🤗_A2R--30B--A3B-ffcc4d)](https://huggingface.co/PleasedPenguin/A2R-30B-A3B)
[![License](https://img.shields.io/badge/License-MIT-555)](LICENSE)

Code for *HEAR Who Said What: Unlocking Speaker-Attributed Reasoning via Counterfactual Voice
Grounding* (EMNLP 2026 Main Conference).

The harness it is built on is **not** specific to HEAR. Every speech LM has its own prompt
format, audio placeholder token, and loading incantation; this hides all of that behind one
JSONL contract, so running twenty models on *your* audio task is a for-loop rather than a month.

---

## Quickstart

```bash
git clone https://github.com/dwsmart32/HEAR && cd HEAR
pip install -r requirements.txt
pip install vllm            # or transformers, depending on the model
```

One line per query:

```json
{"id": "q1", "instruction": "Who speaks second?\n\nOptions:\n(A) ...\n(B) ...", "audio_path": "clip.wav", "ground_truth": "B"}
```

Required fields are `id`, `instruction`, `audio_path`. Add `ground_truth` only if you want the
bundled scorers; relative paths resolve against the JSONL.

```bash
python inference.py --model qwen2_5_omni-7b --input my_task.jsonl --output_dir runs/mine
```

Results are JSON with each model response attached to its input record, so any scorer can read
them. Sweeping models is a shell loop:

```bash
for m in qwen2_5_omni-7b voxtral-mini-3b phi-4-mm-6b; do
  python inference.py --model "$m" --input my_task.jsonl --output_dir runs/mine
done
```

`--validate-input-only` checks every audio path before you spend GPU time.
`examples/run_custom_task.py` builds the JSONL from a folder of clips and runs the sweep for you.

---

## Models

23 models, 26 registry entries, all public Hugging Face ids or API model names. The left column
is what you pass to `--model`. **Start with `qwen2_5_omni-7b`** on a GPU, or `gemini-3-flash`
without one.

| backend | `--model` |
|---|---|
| **vLLM** | `qwen2_5_omni-3b` `qwen2_5_omni-7b` `qwen3-omni-instruct-30b` `qwen3-omni-thinking-30b` `qwen2-audio-7b` `voxtral-mini-3b` `voxtral-small-24b` `phi-4-mm-6b` `midashenglm-7b` `fun-audio-chat-8b` `a2r-30b-a3b` |
| **transformers** | `minicpm-o-2_6-8b` `minicpm-o-4_5-9b` `minicpm-o-4_5-9b-thinking` `kimi-audio-7b-instruct` `gemma4-e2b-it` `gemma4-e4b-it` `audio-flamingo3` |
| **API** | `gpt-4o-audio` `gemini-3-pro` `gemini-3-flash` `gemini-2.5-pro` |
| **external** | `step-audio-2-mini` `step-audio-r1` (bridge not bundled, disabled) |

`requirements.txt` installs the shared core only; add `vllm` or `transformers` as needed. Gemma-4
and Audio-Flamingo 3 want `transformers>=5.0`, Kimi-Audio additionally needs
[`kimia_infer`](https://github.com/MoonshotAI/Kimi-Audio) from source. API models read keys from
`.env` (`cp .env.example .env`).

**Adding a model**: copy the closest entry in `registry.yaml` and change `path`. For vLLM models,
also add the key to `MODEL_FAMILY_MAP` in `src/backends/vllm_handlers/__init__.py`. Qwen-family
fine-tunes need nothing special: the handler reads `config.json` to pick the prompt format, so a
checkpoint under any name is prompted like its base. Override per entry with `system_prompt`.

---

## Layout

```
inference.py        run any model over a JSONL task
registry.yaml       the 26 entries
src/                the harness: runner, four backends, per-family prompt adapters
examples/           run_custom_task.py, a folder of clips to a model sweep
hear/               the bundled benchmark: run_hear.py, the scorers, its prompt audio
```

Nothing under `src/` or in `inference.py` knows what HEAR is.

---

## The HEAR benchmark

**HEAR** asks *who* is speaking, not only what is said: 2,395 questions over 887 multi-party
clips. Half come in counterfactual pairs, where one utterance is re-voiced as another
speaker's voice. Same words, same timing, different speaker, and the answer flips, so a model
reading only the transcript is caught.

The dataset is gated. Once per account: accept the agreement on the
[dataset page](https://huggingface.co/datasets/PleasedPenguin/HEAR), then `hf auth login`.

```bash
python hear/run_hear.py --model qwen2_5_omni-7b        # download, run, score
python hear/eval_hear_vera.py --results runs/hear/results/<model>__hear_input_results.json
```

`hear/eval_hear.py` gives accuracy overall, per axis, per sub-dimension and per category.
`hear/eval_hear_vera.py` gives **Ver-A**, the paper's headline metric, which counts a counterfactual
pair as one unit that scores only when both members are correct: `1,235 items + 580 pairs =
1,815 units`. Transcript-centric models lose 30 to 50 points crossing from per-item accuracy to
Ver-A.

Audio is pre-assembled, so the `audio` column already holds everything the model must hear:
the clip alone for VC/VCD/IR/QR/TR, clip plus a reference voice for VL/VCA, clip plus five
announced voices for CVA.

---

## Also in the release

| | |
|---|---|
| 🤗 [**HEAR**](https://huggingface.co/datasets/PleasedPenguin/HEAR) | 2,395 questions over 887 clips |
| 🤗 `CASH-60K/` in the same repo | 59,762 training queries with counterfactual hard negatives |
| 🤗 `ood_benchmark/` in the same repo | three held-out benchmarks, 1,286 questions |
| 🧠 [**A2R-30B-A3B**](https://huggingface.co/PleasedPenguin/A2R-30B-A3B) | the merged 30B checkpoint |

## Citation

```bibtex
@misc{lee2026hearsaidwhatunlocking,
  title         = {HEAR Who Said What: Unlocking Speaker-Attributed Reasoning
                   via Counterfactual Voice Grounding},
  author        = {Dongwook Lee and Sangkwon Park and Eunwoo Song and Che Hyun Lee
                   and Youngho Cho and Junho Kim and June Young Yi and Heeseung Kim
                   and Sungroh Yoon},
  year          = {2026},
  eprint        = {2608.29120},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2608.29120}
}
```

## License

Code under MIT. HEAR and CASH-60K are CC BY-NC 4.0; the source corpora (AMI, ICSI, VoxMM) keep
their own terms.
