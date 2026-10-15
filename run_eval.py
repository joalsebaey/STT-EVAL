"""CLI: evaluate one or more Whisper models on FLEURS under clean and perturbed conditions."""
import argparse
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from stt_eval.audio import build_conditions
from stt_eval.dataset import load_custom_data
from stt_eval.metrics import top_confusions
from stt_eval.normalize import NormConfig
from stt_eval.pipeline import build_asr, load_fleurs, load_noise_bank, run_conditions, transcribe


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=["openai/whisper-small"])
    p.add_argument("--data", help="path to custom data: CSV/JSON/JSONL file, audio folder, or ZIP archive")
    p.add_argument("--audio-dir", help="base directory for resolving relative audio paths in metadata")
    p.add_argument("--config", default="ar_eg")
    p.add_argument("--split", default="test")
    p.add_argument("--max-samples", type=int, default=100, help="0 = full split")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--language", default="arabic")
    p.add_argument("--conditions", nargs="*", help="subset of condition names; default = all")
    p.add_argument("--noise-dir", help="folder of real noise recordings (wav/flac/mp3/ogg)")
    p.add_argument("--no-auto-language", action="store_true", help="skip the language-detection condition")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("outputs"))
    return p.parse_args()


def save_plots(summary: pd.DataFrame, clean: pd.DataFrame, out: Path, tag: str):
    s = summary.sort_values("wer")
    err = [s.wer - s.wer_ci_low, s.wer_ci_high - s.wer]
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(s) + 1.5))
    ax.barh(s.condition, s.wer, xerr=err, capsize=3)
    ax.set_xlabel("WER (95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(out / f"{tag}_wer_by_condition.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(clean.duration_sec, clean.wer, alpha=0.6)
    ax.set(xlabel="Duration (s)", ylabel="Utterance WER")
    fig.tight_layout()
    fig.savefig(out / f"{tag}_wer_vs_duration.png", dpi=150)
    plt.close(fig)


def clean_slices(clean: pd.DataFrame) -> pd.DataFrame:
    d = clean.assign(duration_bucket=pd.cut(clean.duration_sec, [0, 5, 10, 15, 20, 1e9],
                                            labels=["<5s", "5-10s", "10-15s", "15-20s", ">20s"]))
    rows = []
    cols = ["gender", "duration_bucket"]
    if "dialect" in d.columns and d["dialect"].nunique() > 1:
        cols.append("dialect")
    for col in cols:
        for key, g in d.groupby(col, observed=True):
            rows.append({"slice": col, "value": str(key), "n": len(g),
                         "wer": g.w_err.sum() / g.w_ref_len.sum(), "cer": g.c_err.sum() / g.c_ref_len.sum()})
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    import torch

    device = 0 if torch.cuda.is_available() else -1
    norm_cfg = NormConfig()
    noise_bank = load_noise_bank(args.noise_dir)
    conditions = build_conditions(noise_bank)
    if args.conditions:
        conditions = {k: conditions[k] for k in args.conditions}
    items = load_fleurs(args.config, args.split, args.max_samples or None, args.seed)
    if args.data:
        items = load_custom_data(
            args.data, audio_dir=args.audio_dir,
            max_samples=args.max_samples or None, seed=args.seed,
        )
    args.out.mkdir(parents=True, exist_ok=True)

    for model_id in args.models:
        tag = model_id.split("/")[-1]
        out = args.out / tag
        out.mkdir(exist_ok=True)
        asr = build_asr(model_id, device)
        auto_fn = None if args.no_auto_language else (lambda arrays: transcribe(asr, arrays, args.batch_size, None))
        summary, details = run_conditions(
            items,
            lambda arrays: transcribe(asr, arrays, args.batch_size, args.language),
            conditions, norm_cfg, args.batch_size, args.seed, args.bootstrap,
            auto_fn,
        )
        summary.to_csv(out / "summary_by_condition.csv", index=False, encoding="utf-8-sig")
        for name, df in details.items():
            df.drop(columns=["ref", "hyp"]).to_csv(out / f"utterances_{name}.csv", index=False, encoding="utf-8-sig")

        if "clean" in details:
            clean = details["clean"]
            clean_slices(clean).to_csv(out / "slices_clean.csv", index=False, encoding="utf-8-sig")
            if "dialect" in clean.columns and clean["dialect"].nunique() > 1:
                from stt_eval.evaluator import dialect_summary
                dialect_summary(clean).to_csv(out / "dialect_wer.csv", index=False, encoding="utf-8-sig")
            clean.nlargest(30, "wer")[["row_index", "reference_raw", "prediction_raw", "wer"]].to_csv(
                out / "worst_30_clean.csv", index=False, encoding="utf-8-sig")
            top_confusions(clean.ref, clean.hyp).to_csv(out / "top_confusions_clean.csv", index=False, encoding="utf-8-sig")
            save_plots(summary, clean, out, tag)

        report = {"model": model_id, "dataset": f"google/fleurs:{args.config}:{args.split}",
                  "num_examples": len(items), "language_arg": args.language, "seed": args.seed,
                  "normalization": norm_cfg.describe(), "python": platform.python_version(),
                  "device": "cuda" if device == 0 else "cpu", "summary": summary.to_dict("records")}
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(summary[["condition", "wer", "wer_ci_low", "wer_ci_high", "cer", "ins_rate", "rtf"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
