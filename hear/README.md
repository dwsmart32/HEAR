# HEAR, as one bundled task

Everything HEAR-specific lives here. The harness in `../src/` and `../inference.py` does not
import any of it, so you can ignore this folder entirely if you came for the inference tool.

```
run_hear.py                    download from the Hub, run a model, score it
eval_hear.py                   the paper's scorer: overall, per axis, per sub-dimension
eval_hear_vera.py              Ver-A, the headline metric (a pair counts as one unit)
eval_hear_lenient.py           appendix variant, looser answer extraction
eval_hear_reasoning_strict.py  appendix variant, Reasoning pairs only
assets/prompt_audio/           spoken option letters A-E and bridges used to assemble CVA audio
```

```bash
python hear/run_hear.py --model qwen2_5_omni-7b
python hear/eval_hear_vera.py --results runs/hear/results/<model>__hear_input_results.json
```

The dataset is gated: accept the agreement on the
[dataset page](https://huggingface.co/datasets/PleasedPenguin/HEAR), then `hf auth login`.

Scoring your own task instead? Use `../examples/run_custom_task.py`.
