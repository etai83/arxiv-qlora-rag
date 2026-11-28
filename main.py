"""Unified CLI entrypoint for the arXiv temporal RAG & QLoRA evaluation project."""

from __future__ import annotations

import argparse
import json
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="arXiv cs.CL temporal split, FAISS RAG, baselines, and LLM-as-a-Judge CLI."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("fetch", help="Fetch date-split cs.CL static/live corpus from arXiv API.")

    ingest_p = sub.add_parser("ingest", help="Embed corpus/live.jsonl into FAISS index.")
    ingest_p.add_argument("--live-path", default="corpus/live.jsonl")
    ingest_p.add_argument("--index-dir", default="data/index")

    ret_p = sub.add_parser("retrieve", help="Query the live FAISS index with optional date filter.")
    ret_p.add_argument("--query", required=True)
    ret_p.add_argument("--k", type=int, default=4)
    ret_p.add_argument("--after", default=None, help="Inclusive YYYY-MM-DD minimum date filter.")
    ret_p.add_argument("--index-dir", default="data/index")

    base_p = sub.add_parser("baselines", help="Run zero-shot and RAG baselines via Ollama.")
    base_p.add_argument("--golden-path", default="eval/golden.jsonl")
    base_p.add_argument("--index-dir", default="data/index")
    base_p.add_argument("--out-dir", default="eval")
    base_p.add_argument("--k", type=int, default=4)
    base_p.add_argument("--model", default="llama3.2:3b")
    base_p.add_argument("--zs-out", default=None)
    base_p.add_argument("--rag-out", default=None)

    judge_p = sub.add_parser("judge", help="Score baseline outputs with Qwen2.5-7B judge.")
    judge_p.add_argument("--zs-path", default="eval/outputs_zero_shot.jsonl")
    judge_p.add_argument("--rag-path", default="eval/outputs_rag.jsonl")
    judge_p.add_argument("--out-path", default="eval/judge_scores.jsonl")
    judge_p.add_argument("--model", default="qwen2.5:7b")

    metrics_p = sub.add_parser("metrics", help="Aggregate results summary and judge-vs-human agreement.")
    metrics_p.add_argument("--export-blind", action="store_true")
    metrics_p.add_argument("--zs-path", default="eval/outputs_zero_shot.jsonl")
    metrics_p.add_argument("--rag-path", default="eval/outputs_rag.jsonl")
    metrics_p.add_argument("--out-path", default="eval/blind_sample_20.jsonl")
    metrics_p.add_argument("--seed", type=int, default=42)
    metrics_p.add_argument("--judge-path", default="eval/judge_scores.jsonl")
    metrics_p.add_argument("--human-path", default="eval/human_scores_20.jsonl")
    metrics_p.add_argument("--out-dir", default="eval")
    metrics_p.add_argument("--json-name", default="results_summary.json")
    metrics_p.add_argument("--md-name", default="results_summary.md")

    compare_p = sub.add_parser("compare-judges", help="Statistically compare two judge runs.")
    compare_p.add_argument("--base-judge", default="eval/judge_scores.jsonl")
    compare_p.add_argument("--cand-judge", default="eval/judge_scores_qwen35_9b.jsonl")
    compare_p.add_argument("--human-path", default="eval/human_scores_20.jsonl")
    compare_p.add_argument("--zs-path", default="eval/outputs_zero_shot.jsonl")
    compare_p.add_argument("--rag-path", default="eval/outputs_rag.jsonl")
    compare_p.add_argument("--base-label", default="qwen2.5:7b")
    compare_p.add_argument("--cand-label", default="qwen3.5:9b-mlx")
    compare_p.add_argument("--out-json", default="eval/judge_comparison.json")
    compare_p.add_argument("--out-md", default="eval/judge_comparison.md")

    book_p = sub.add_parser("book", help="Build interactive Tailwind/shadcn/Plotly experiment_book.html.")
    book_p.add_argument("--root", default=".")
    book_p.add_argument("--out-path", default="experiment_book.html")

    prep_p = sub.add_parser("prepare-qlora", help="Prepare QLoRA training/validation/test JSONL and config.")
    prep_p.add_argument("--static-path", default="corpus/static.jsonl")
    prep_p.add_argument("--out-dir", default="data/qlora")
    prep_p.add_argument("--model", default="qwen3.5:9b")
    prep_p.add_argument("--max-workers", type=int, default=4)
    prep_p.add_argument("--seed", type=int, default=42)
    prep_p.add_argument("--skip-distill", action="store_true")

    train_p = sub.add_parser("train-qlora", help="Run MLX QLoRA training, optional adapter fusing, and Modelfile build.")
    train_p.add_argument("--config", default="data/qlora/lora_config.yaml")
    train_p.add_argument("--python-bin", default=None)
    train_p.add_argument("--iters", type=int, default=600)
    train_p.add_argument("--fuse", action="store_true")
    train_p.add_argument("--build-modelfile", action="store_true")

    cmp_models_p = sub.add_parser("compare-models", help="Statistically compare base vs fine-tuned model evaluations.")
    cmp_models_p.add_argument("--base-judge", default="eval/judge_scores_qwen35_9b.jsonl")
    cmp_models_p.add_argument("--ft-judge", default="eval/judge_scores_ft_qwen35_9b.jsonl")
    cmp_models_p.add_argument("--sec-base-judge", "--secondary-base-judge", dest="sec_base_judge", default="eval/judge_scores.jsonl")
    cmp_models_p.add_argument("--sec-ft-judge", "--secondary-ft-judge", dest="sec_ft_judge", default="eval/judge_scores_ft_qwen25_7b.jsonl")
    cmp_models_p.add_argument("--out-json", default="eval/model_comparison.json")
    cmp_models_p.add_argument("--out-md", default="eval/model_comparison.md")
    cmp_models_p.add_argument("--base-label", default="llama3.2:3b")
    cmp_models_p.add_argument("--ft-label", default="llama3.2-qlora:3b")

    return parser


def main(argv: list[str] | None = None) -> None:
    """Dispatch CLI subcommands."""
    args = _build_parser().parse_args(argv)

    if args.command == "fetch":
        from src.fetch_corpus import main as fetch_main

        fetch_main()
    elif args.command == "ingest":
        from src.ingest import ingest

        stats = ingest(live_path=args.live_path, index_dir=args.index_dir)
        print(json.dumps(stats, indent=2))
    elif args.command == "retrieve":
        from src.retriever import retrieve

        hits = retrieve(args.query, k=args.k, after=args.after, index_dir=args.index_dir)
        print(json.dumps(hits, indent=2, ensure_ascii=False))
    elif args.command == "baselines":
        from src.baselines import run_baselines

        zs_out, rag_out = run_baselines(
            golden_path=args.golden_path,
            index_dir=args.index_dir,
            out_dir=args.out_dir,
            k=args.k,
            model=args.model,
            zs_out=args.zs_out,
            rag_out=args.rag_out,
        )
        print(f"Wrote {zs_out} and {rag_out}")
    elif args.command == "judge":
        from src.judge import run_judge

        out = run_judge(
            zs_path=args.zs_path,
            rag_path=args.rag_path,
            out_path=args.out_path,
            model=args.model,
        )
        print(f"Wrote {out}")
    elif args.command == "metrics":
        from src.metrics import export_blind_sample, write_report

        if args.export_blind:
            out = export_blind_sample(
                zs_path=args.zs_path,
                rag_path=args.rag_path,
                out_path=args.out_path,
                seed=args.seed,
            )
            print(f"Wrote blind sample to {out}")
            return
        human_path = (
            None
            if args.human_path is None or args.human_path.strip().lower() in {"", "none"}
            else args.human_path
        )
        json_path, md_path = write_report(
            judge_path=args.judge_path,
            zs_path=args.zs_path,
            rag_path=args.rag_path,
            human_path=human_path,
            out_dir=args.out_dir,
            json_name=args.json_name,
            md_name=args.md_name,
        )
        print(f"Wrote {json_path} and {md_path}")
    elif args.command == "compare-judges":
        from src.compare_judges import write_comparison

        json_path, md_path = write_comparison(
            base_judge_path=args.base_judge,
            cand_judge_path=args.cand_judge,
            human_path=args.human_path,
            zs_path=args.zs_path,
            rag_path=args.rag_path,
            out_json=args.out_json,
            out_md=args.out_md,
            base_label=args.base_label,
            cand_label=args.cand_label,
        )
        print(f"Wrote comparison JSON to {json_path} and Markdown to {md_path}")
    elif args.command == "book":
        from src.build_book import build_book

        out = build_book(root=args.root, out_path=args.out_path)
        print(f"Wrote {out}")
    elif args.command == "prepare-qlora":
        from src.prepare_qlora_data import prepare_qlora

        res = prepare_qlora(
            static_path=args.static_path,
            out_dir=args.out_dir,
            model=args.model,
            max_workers=args.max_workers,
            seed=args.seed,
            skip_distill=args.skip_distill,
        )
        print(json.dumps(res, indent=2, default=str))
    elif args.command == "train-qlora":
        from src.train_qlora import build_modelfile, run_training

        res = run_training(
            config_path=args.config,
            python_bin=args.python_bin,
            iters=args.iters,
            fuse=args.fuse,
        )
        if args.build_modelfile and not res.get("modelfile_path"):
            res["modelfile_path"] = str(build_modelfile())
        print(json.dumps(res, indent=2, default=str))
    elif args.command == "compare-models":
        from src.compare_models import write_model_comparison

        res = write_model_comparison(
            base_judge_path=args.base_judge,
            ft_judge_path=args.ft_judge,
            secondary_base_judge_path=args.sec_base_judge,
            secondary_ft_judge_path=args.sec_ft_judge,
            out_json=args.out_json,
            out_md=args.out_md,
            base_label=args.base_label,
            ft_label=args.ft_label,
        )
        if isinstance(res, tuple) and len(res) == 2:
            print(f"Wrote model comparison JSON to {res[0]} and Markdown to {res[1]}")
        else:
            print(json.dumps(res, indent=2, default=str))
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
