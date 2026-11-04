"""`miroir vlm`: vision models asked about query photos (steps H1, H4)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from miroir.eval.h4 import Loaded
    from miroir.jobs.h4_agent import Outcome, Scenario

from miroir.cli.common import key_usage
from miroir.config import get_settings

app = typer.Typer(
    no_args_is_help=True,
    help="Step H1: ask a vision model about query photos, search its answer.",
)

DEFAULT_GALLERY = "screening-20k-publishable-fetched"
DEFAULT_SETS = ("T1", "T3", "T5")


@app.command("ask")
def vlm_ask(
    model: Annotated[str, typer.Option(help="qwen3-vl-30b-a3b or gemini-3.5-flash-lite.")],
    *,
    test_set: Annotated[
        list[str] | None, typer.Option("--set", help="Tune half to ask about; repeat it.")
    ] = None,
    gallery: Annotated[str, typer.Option(help="Keep queries whose image is here.")] = (
        DEFAULT_GALLERY
    ),
    probe: Annotated[
        bool, typer.Option("--probe", help="Ask about 100 query images; print the projection.")
    ] = False,
    max_usd: Annotated[float, typer.Option(help="Refuse or stop a run past this.")] = 1.0,
    batch: Annotated[
        bool,
        typer.Option(
            "--batch/--no-batch",
            help="Batch API when the model has one. --no-batch: standard endpoint, full price, "
            "for images the batch could not fetch.",
        ),
    ] = True,
) -> None:
    """Ask a vision model about each publishable query image. Paid: probe first."""
    from miroir.jobs import describe  # noqa: PLC0415
    from miroir.jobs.factory import open_describer  # noqa: PLC0415

    settings = get_settings()
    sets = tuple(test_set or DEFAULT_SETS)
    ids = describe.query_image_ids(settings, sets, gallery)
    with open_describer(settings, model, batch=batch) as describer:
        if probe:
            result = describe.probe(describer, ids, "+".join(sets) + "-tune")
            typer.echo(
                f"{model} probe: {result.images} images cost ${result.cost_usd:.4f}. "
                f"Projected for {len(ids)} query images: ${result.projected_usd(len(ids)):.3f}"
            )
            return
        done = describe.run(describer, ids, max_usd=max_usd)
    typer.echo(f"{model}: {done.answered} new answers, ${done.cost_usd:.4f}, skipped {done.skips}")


@app.command("h4")
def vlm_h4(
    *,
    model: Annotated[str, typer.Option(help="Vision model leading the search.")] = (
        "mistral-large-4"
    ),
    scenario: Annotated[
        list[str] | None,
        typer.Option("--scenario", help="H1, S1, S2, S3, S4; repeat it. Default: all."),
    ] = None,
    per_set: Annotated[
        int, typer.Option(help="Queries drawn from each of T1, T2, T3, T5 (seeded).")
    ] = 5,
    every_query: Annotated[
        bool,
        typer.Option("--all", help="Every tune query a closed model may see; writes results."),
    ] = False,
    photos: Annotated[
        bool,
        typer.Option(
            "--photos/--per-set",
            help="The 10-photo test (T1 2, T2 3 with 2 placed, T3 2, T5 3), or --per-set.",
        ),
    ] = True,
    label: Annotated[str | None, typer.Option(help="Report folder name.")] = None,
    max_usd: Annotated[float, typer.Option(help="Stop once this run has paid this much.")] = 2.0,
    workers: Annotated[int, typer.Option(help="Queries asked in parallel.")] = 4,
) -> None:
    """Step H4: a vision model leads the search (S1 plan, S2 plan + review, S3 free loop).

    Paid. Replies are saved and replayed, so a rerun costs nothing. Without --all, a smoke
    test: --per-set queries from each set, no result file.
    """
    from miroir.eval import h4  # noqa: PLC0415
    from miroir.jobs.factory import open_agent_caller  # noqa: PLC0415
    from miroir.jobs.h4_agent import Scenario  # noqa: PLC0415
    from miroir.jobs.spend import Budget, SpendLedger, check_headroom  # noqa: PLC0415

    settings = get_settings()
    scenarios = [Scenario(s) for s in (scenario or [s.value for s in Scenario])]
    check_headroom(
        Budget.from_settings(settings), SpendLedger(settings.spend_ledger), key_usage(), max_usd
    )
    loaded = h4.load_world(settings)
    if every_query:
        queries = h4.pick(loaded, None)
    elif photos:
        queries = h4.pick_photos(loaded)
    else:
        queries = h4.pick(loaded, per_set)
    name = label or ("full" if every_query else "photos-10" if photos else f"smoke-{per_set}")
    typer.echo(f"H4 {model}: {len(queries)} queries, {', '.join(s.value for s in scenarios)}")
    with open_agent_caller(settings, model, limit_usd=max_usd) as caller:
        caller.check_images([q.query.image_id for q in queries])
        outcomes = h4.run(caller, loaded, queries, scenarios, workers=workers)
        new_spend = caller.spent_usd
    rows, extra = h4.score(loaded, outcomes)
    counts = {s: len(qs) for s, qs in loaded.sets.items()}
    costs = h4.cost_report(outcomes, counts)
    folder = settings.h4_report_dir(name)
    h4.write_traces(folder / "traces.jsonl", outcomes)
    report = {
        "model": model,
        "queries": len(queries),
        "full_counts": counts,
        "new_spend_usd": new_spend,
        "costs": costs,
        "scores": [
            {
                "run": r.name,
                "set": r.set_id,
                "queries": r.score.queries,
                "hit@1": r.score.hit_at_1.value,
                "hit@10": r.score.hit_at_10.value,
                "hit@1 - G5": r.vs_g5["hit@1"].diff.value,
                "hit@10 - G5": r.vs_g5["hit@10"].diff.value,
                **(
                    {
                        "hit@1 - S3": r.vs_s3["hit@1"].diff.value,
                        "hit@10 - S3": r.vs_s3["hit@10"].diff.value,
                    }
                    if r.vs_s3 is not None
                    else {}
                ),
            }
            for r in rows
        ],
        **extra,
    }
    (folder / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if photos and not every_query:
        _echo_photos(loaded, outcomes, folder)
    else:
        for line in [*h4.table(rows), *h4.table_s4(rows)]:
            typer.echo(line)
    for s, c in costs.items():
        typer.echo(f"{s}: {json.dumps(c, ensure_ascii=False)}")
    typer.echo(f"paid this run: ${new_spend:.4f}; traces and report in {folder}")
    if every_query:
        from miroir.jobs.h4_agent import OPTIONS as AGENT_OPTIONS  # noqa: PLC0415

        option = AGENT_OPTIONS[model]
        for path in h4.write_results(
            loaded,
            rows,
            outcomes,
            model=option.model,
            slug=option.slug,
            h3_weight=extra.get("h3_a7_weight"),
        ):
            typer.echo(f"wrote {path}")


def _echo_photos(loaded: Loaded, got: Mapping[Scenario, Sequence[Outcome]], folder: Path) -> None:
    """Per photo: each run's rank of the right answer, cost, and short S3 and S4 traces."""
    from miroir.eval import h4  # noqa: PLC0415
    from miroir.jobs.h4_agent import Scenario  # noqa: PLC0415

    ranks = h4.photo_ranks(loaded, got)
    costs = h4.photo_costs(got)
    names = list(next(iter(ranks.values())))
    typer.echo("rank of the right answer (1 = first)")
    typer.echo(
        f"{'set':<4}{'query':<30}{'bucket':<18}{'gps':<5}" + "".join(f"{n:>13}" for n in names)
    )
    for qid, by_run in ranks.items():
        q = loaded.world.queries[qid]
        gps = "yes" if loaded.facts[qid].position else ""
        typer.echo(
            f"{q.set_id:<4}{qid:<30}{q.bucket:<18}{gps:<5}"
            + "".join(f"{by_run[n]:>13}" for n in names)
        )
    typer.echo("cost per photo: $ (tokens in / out)")
    for qid, by_s in costs.items():
        cells = "  ".join(
            f"{s} ${c['cost_usd']:.4f} ({c['prompt_tokens']:.0f}/{c['completion_tokens']:.0f})"
            for s, c in sorted(by_s.items())
        )
        typer.echo(f"{qid:<30}{cells}")
    lines: list[str] = []
    for scenario in (Scenario.S3, Scenario.S4):
        for o in got.get(scenario, ()):
            lines += h4.short_trace(o, loaded.world.describe)
    (folder / "traces-short.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (folder / "photos.json").write_text(
        json.dumps({"ranks": ranks, "costs": costs}, indent=2, ensure_ascii=False) + "\n"
    )
    typer.echo(f"short traces: {folder / 'traces-short.txt'}")
