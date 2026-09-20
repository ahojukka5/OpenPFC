#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Archive compact tournament evidence; replay summaries without raw profiles.

Extract: --root CAMPAIGN --out ARCHIVE_DIRECTORY
Replay:  --replay ARCHIVE_DIRECTORY
Large profiles are omitted with hashes; exact reduced inputs, original metadata,
logs and scripts remain committed. Timing replay does not promise identical
measurements on a later machine allocation.
"""
import argparse
import gzip
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path

import heffte_protocol_tournament as tournament


def digest(data):
    return hashlib.sha256(data).hexdigest()


def extract(root):
    artifacts = {}

    def capture(path):
        data = path.read_bytes()
        key = digest(data)
        artifacts[key] = data.decode()
        return key

    rows = tournament.collect_root(str(root), warmup=1)
    records = []
    for row in rows:
        directory = Path(row["scratch"])
        run = directory.parent
        meta = tournament._meta_map(str(run / "run_meta.txt"))
        entry = {"row": row, "files": {}, "run_directory": run.name,
                 "steps": int(meta["steps"]), "warmup": int(meta["warmup"]),
                 "dt": meta["dt"]}
        for path in [run / "run_meta.txt", run / "heffte_protocol_tournament.sbatch",
                     directory / "run.log", directory / "admit.txt"]:
            if path.exists():
                entry["files"][str(path.relative_to(root))] = capture(path)
        profile = directory / "timing_profile.json"
        if row["admit"] == "ok":
            data = profile.read_bytes()
            document = json.loads(data)
            names = document["frame_metric_names"]
            si, wi = names.index("step"), names.index("wall_step")
            samples, old_samples = [], []
            frame_counts, step_ranges = [], []
            for rank in document["ranks"]:
                frames = rank["frames"]
                steps = [frame["scalars"][si] for frame in frames]
                if steps != list(range(entry["warmup"], entry["steps"])):
                    raise ValueError(f"unexpected exported steps in {profile}")
                values = [frame["scalars"][wi] for frame in frames]
                if not all(math.isfinite(x) and x > 0 for x in values):
                    raise ValueError(f"invalid timing in {profile}")
                samples.extend(values)
                old_samples.extend(values[1:])
                frame_counts.append(len(frames))
                step_ranges.append([min(steps), max(steps)])
            if len(document["ranks"]) != int(row["ranks"]):
                raise ValueError(f"rank mismatch in {profile}")
            ordered = sorted(samples)
            count = len(ordered)
            middle = ordered[(count - 1)//2:count//2 + 1]
            median = statistics.mean(middle)
            if f"{median:.8f}" != row["wall_step_s"]:
                raise ValueError(f"independent median disagrees in {profile}")
            entry["timing"] = {
                "profile_sha256": digest(data), "profile_bytes": len(data),
                "sample_count": count, "rank_count": len(document["ranks"]),
                "frames_per_rank": sorted(set(frame_counts)),
                "absolute_step_ranges": sorted(set(map(tuple, step_ranges))),
                "central_order_statistics": middle, "minimum": min(samples),
                "maximum": max(samples), "median": median,
                "previous_double_warmup_median": statistics.median(old_samples),
                "statistic": "pooled rank/frame median of all exported steps (absolute step >= 1)",
            }
        records.append(entry)
    top = {p.name: capture(p) for p in [root / "campaign_meta.txt", root / "failures.txt"]
           if p.exists()}
    jobs = sorted(set(str(row["job"]) for row in rows))
    scheduler = subprocess.check_output([
        "sacct", "-X", "-j", ",".join(jobs), "--parsable2", "--noheader",
        "--format=JobID,Account,State,ExitCode,Elapsed,NNodes,NTasks,AllocTRES"], text=True)
    accounts = {line.split("|")[0]: line.split("|")[1] for line in scheduler.splitlines()}
    for row in rows:
        if accounts.get(str(row["job"])) != row["account"]:
            raise ValueError(f"scheduler account mismatch for {row['job']}")
    return {"records": records, "artifacts_by_sha256": artifacts,
            "campaign_files": top, "scheduler_records": scheduler,
            "collector_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "collector_sha256": digest(Path(tournament.__file__).read_bytes()),
            "extractor_sha256": digest(Path(__file__).read_bytes()),
            "raw_root": str(root), "frozen_warmup": 1}


def replay(directory):
    bundle = json.loads(gzip.decompress((directory / "records.json.gz").read_bytes()))
    for sha, content in bundle["artifacts_by_sha256"].items():
        if digest(content.encode()) != sha:
            raise ValueError("archive artifact hash mismatch")
    rows = [record["row"] for record in bundle["records"]]
    for record in bundle["records"]:
        if "timing" in record:
            timing = record["timing"]
            median = statistics.mean(timing["central_order_statistics"])
            if median != timing["median"] or f"{median:.8f}" != record["row"]["wall_step_s"]:
                raise ValueError("compact reduction mismatch")
    tournament.write_csv(str(directory / "all-runs.csv"), tournament.RUN_FIELDS, rows)
    # Keep the frozen 20-step/one-warmup protocol separate from longer references.
    compliant = [record["row"] for record in bundle["records"]
                 if record["row"]["account"] == "project_462001245"
                 and record["steps"] == 20 and record["warmup"] == 1
                 and float(record["dt"]) == 0.01]
    tournament.write_csv(str(directory / "campaign-runs.csv"), tournament.RUN_FIELDS, compliant)
    scaled = tournament.analyze(compliant)
    tournament.write_csv(str(directory / "scaling.csv"), tournament.SCALE_FIELDS, scaled)
    tournament.write_markdown(str(directory / "scaling.md"), scaled,
                              tournament.crossover_notes(scaled))
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()
    if args.replay:
        replay(args.replay)
    elif args.root and args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        bundle = extract(args.root)
        data = (json.dumps(bundle, sort_keys=True, indent=2) + "\n").encode()
        (args.out / "records.json.gz").write_bytes(gzip.compress(data, mtime=0))
        replay(args.out)
    else:
        parser.error("use --root and --out, or --replay")


if __name__ == "__main__":
    main()
