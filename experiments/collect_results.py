#!/usr/bin/env python3
"""Collect the branched-start experiments into tables (stdlib only).

    python experiments/collect_results.py [--work DIR]

Reads   $WORK/logs/<setting>-from<start>-<arm>-s<seed>.log   (in-training validation, "step:N - val/...:x" lines)
        $WORK/evals/<run>/unseen_step<k>.log                  (ALFWorld unseen split, 134 games, optional)
        $WORK/evals/<run>/ws500_step<k>.log                   (WebShop, all 500 test goals in one process)
        $WORK/evals/start-<setting>-<start>/*.log             (the start point, evaluated once)
        $WORK/done/<run>                                      (eval=<END steps> of the run)
Writes  $WORK/results/runs.csv      one row per run x END step
        $WORK/results/curves.csv    training success, in-training validation and distillation diagnostics per step
        $WORK/results/summary.md    per (setting, start, step): mean +- std over seeds and the paired difference
                                    to the grpo arm (same setting, start and seed) with a 95% t-interval
"""
import argparse
import csv
import math
import os
import re
import statistics as st
from collections import defaultdict

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP_LINE = re.compile(r"step:(\d+) - (.*)")
SETTINGS = ["3b-alf", "3b-ws", "17b-alf", "17b-ws"]
NAME_RE = re.compile(r"^(%s)-from(\d+)-(.+)-s(\d+)$" % "|".join(map(re.escape, SETTINGS)))
ALF_TYPES = [("pick_and_place", "Pick"), ("look_at_obj_in_light", "Look"),
             ("pick_clean_then_place_in_recep", "Clean"), ("pick_heat_then_place_in_recep", "Heat"),
             ("pick_cool_then_place_in_recep", "Cool"), ("pick_two_obj_and_place", "Pick2")]
WS_SCORE = "val/webshop_task_score (not success_rate)"
ARM_ORDER = ["grpo", "pr-opd", "pr-opd-noproj", "token-opd", "skillsd-self", "pr-opd-shuf"]
# two-sided 95% t quantiles, df = 1..30
T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145,
        2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def parse_fields(text):
    out = {}
    for f in text.split(" - "):
        if ":" not in f:
            continue
        k, v = f.rsplit(":", 1)
        try:
            out[k.strip()] = float(v.strip())
        except ValueError:
            pass
    return out


def parse_log(path, val_only=False):
    """{step: {metric: value}}; later lines override earlier ones (resumed runs)."""
    steps = defaultdict(dict)
    with open(path, errors="replace") as f:
        for line in f:
            m = STEP_LINE.search(ANSI.sub("", line))
            if m:
                fields = parse_fields(m.group(2))
                if val_only:
                    fields = {k: v for k, v in fields.items() if k.startswith("val/")}
                steps[int(m.group(1))].update(fields)
    return steps


def last_val(path):
    """Last value of every val/ metric in a val_only log."""
    vals = {}
    for _, v in sorted(parse_log(path, val_only=True).items()):
        vals.update(v)
    return vals


def pct(x):
    return None if x is None else 100 * x


def mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else float("nan")), len(xs)


def ci95(diffs):
    m, sd, n = mean_sd(diffs)
    if n < 2:
        return m, None, None, n
    h = T975[min(n - 1, len(T975)) - 1] * sd / math.sqrt(n)
    return m, m - h, m + h, n


def fmt(x, nd=1):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def end_steps_of(done_path):
    if os.path.exists(done_path):
        for line in open(done_path):
            if line.startswith("eval="):
                return [int(x) for x in line[5:].split()]
    return None


def env_metrics(env, val, ev_dir, k):
    """Metrics of one run at global step k: val = in-training validation at k, ev_dir = final eval logs."""
    r = {}
    if env == "alfworld":
        r["seen140"] = pct(val.get("val/success_rate"))
        for key, short in ALF_TYPES:
            r[f"seen_{short}"] = pct(val.get(f"val/{key}_success_rate"))
        p = os.path.join(ev_dir, f"unseen_step{k}.log")
        if os.path.exists(p):
            r["unseen134"] = pct(last_val(p).get("val/success_rate"))
    else:
        p = os.path.join(ev_dir, f"ws500_step{k}.log")
        if os.path.exists(p):
            e = last_val(p)   # the trainer already averaged the 4 equal batches of 125 goals
            r["ws500_acc"], r["ws500_score"] = pct(e.get("val/success_rate")), pct(e.get(WS_SCORE))
        r["ws125_acc_intrain"] = pct(val.get("val/success_rate"))
        r["ws125_score_intrain"] = pct(val.get(WS_SCORE))
    return r


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--work", default=os.environ.get("WORK", os.path.join(os.path.dirname(here), "work")))
    a = ap.parse_args()
    logs, evals, done = (os.path.join(a.work, d) for d in ("logs", "evals", "done"))
    res = os.path.join(a.work, "results")
    os.makedirs(res, exist_ok=True)

    runs, curves = [], []
    for fn in sorted(os.listdir(logs)) if os.path.isdir(logs) else []:
        if not fn.endswith(".log"):
            continue
        name = fn[:-4]
        m = NAME_RE.match(name)
        if not m:
            continue
        setting, start, arm, seed = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
        env = "alfworld" if setting.endswith("alf") else "webshop"
        tr = {g: v for g, v in parse_log(os.path.join(logs, fn)).items() if g > start}
        for g in sorted(tr):
            v = tr[g]
            row = {"setting": setting, "start": start, "arm": arm, "seed": seed, "global_step": g}
            if "episode/success_rate" in v:
                row["train_success"] = 100 * v["episode/success_rate"]
            if "val/success_rate" in v:
                row["val_success"] = 100 * v["val/success_rate"]
                if WS_SCORE in v:
                    row["val_score"] = 100 * v[WS_SCORE]
            for k, x in v.items():
                if "oprd/" in k or k.endswith("sdar/loss") or "sdl_loss" in k:
                    row[k] = x
            if len(row) > 5:
                curves.append(row)
        ends = end_steps_of(os.path.join(done, name))
        if ends is None:   # unfinished run: report the validation passes at the usual END steps reached so far
            ends = [g for g in (150, 200) if g in tr]
        for k in ends:
            if k not in tr or "val/success_rate" not in tr[k]:
                continue
            r = {"setting": setting, "start": start, "arm": arm, "seed": seed, "env": env, "global_step": k,
                 "finished": os.path.exists(os.path.join(done, name))}
            r.update(env_metrics(env, tr[k], os.path.join(evals, name), k))
            runs.append(r)

    # start points (evaluated once per setting and start by eval_start.sh)
    starts = {}
    for d in sorted(os.listdir(evals)) if os.path.isdir(evals) else []:
        m = re.match(r"^start-(%s)-(\d+)$" % "|".join(map(re.escape, SETTINGS)), d)
        if not m:
            continue
        setting, s = m.group(1), int(m.group(2))
        ed = os.path.join(evals, d)
        r = {}
        if os.path.exists(os.path.join(ed, "seen140.log")):
            e = last_val(os.path.join(ed, "seen140.log"))
            r["seen140"] = pct(e.get("val/success_rate"))
            for key, short in ALF_TYPES:
                r[f"seen_{short}"] = pct(e.get(f"val/{key}_success_rate"))
        if os.path.exists(os.path.join(ed, "unseen134.log")):
            r["unseen134"] = pct(last_val(os.path.join(ed, "unseen134.log")).get("val/success_rate"))
        if os.path.exists(os.path.join(ed, "ws500.log")):
            e = last_val(os.path.join(ed, "ws500.log"))
            r["ws500_acc"], r["ws500_score"] = pct(e.get("val/success_rate")), pct(e.get(WS_SCORE))
        if os.path.exists(os.path.join(ed, "ws125.log")):
            e = last_val(os.path.join(ed, "ws125.log"))
            r["ws125_acc_intrain"], r["ws125_score_intrain"] = pct(e.get("val/success_rate")), pct(e.get(WS_SCORE))
        r = {k: v for k, v in r.items() if v is not None}
        if r:
            starts[(setting, s)] = r
            runs.append({"setting": setting, "start": s, "arm": "start", "seed": 0,
                         "env": "alfworld" if setting.endswith("alf") else "webshop", "global_step": s, **r})

    def write_csv(path, rows):
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    write_csv(os.path.join(res, "runs.csv"), runs)
    write_csv(os.path.join(res, "curves.csv"), curves)

    # ---------------- summary ----------------
    metrics = {"alfworld": [("seen140", "ALFWorld seen (140)"), ("unseen134", "ALFWorld unseen (134)")]
               + [(f"seen_{s}", f"seen · {s}") for _, s in ALF_TYPES],
               "webshop": [("ws500_acc", "WebShop Acc (500)"), ("ws500_score", "WebShop Score (500)"),
                           ("ws125_acc_intrain", "Acc (in-training, goals 0–124)")]}
    by = defaultdict(dict)   # (setting, start, step) -> arm -> seed -> row
    for r in runs:
        if r["arm"] != "start":
            by[(r["setting"], r["start"], r["global_step"])].setdefault(r["arm"], {})[r["seed"]] = r
    L = ["# Results", "",
         "Every run resumes the GRPO trunk checkpoint at its start step (weights + optimizer + step counter; KL reference "
         "= base model). Mean ± std over training seeds; Δ = paired difference to `grpo` with the same setting, start "
         "and seed (95% t-interval). ALFWorld seen = all 140 in-distribution games once each (in-training validation "
         "at that step); unseen = 134 out-of-distribution games; WebShop = all 500 test goals once each. "
         "Runs marked * are not finished.", ""]
    for (setting, s, k) in sorted(by, key=lambda x: (SETTINGS.index(x[0]), x[1], x[2])):
        arms = by[(setting, s, k)]
        env = "alfworld" if setting.endswith("alf") else "webshop"
        L += [f"## {setting} — start GRPO@{s} → global step {k}", "",
              "| arm | metric | mean ± std | n | per seed | Δ vs grpo (95% CI) |", "|---|---|---|---|---|---|"]
        for key, label in metrics[env]:
            if key in starts.get((setting, s), {}):
                L.append(f"| start (GRPO@{s}) | {label} | {fmt(starts[(setting, s)][key])} | 1 | – | – |")
        base = arms.get("grpo", {})
        for arm in sorted(arms, key=lambda x: ARM_ORDER.index(x) if x in ARM_ORDER else 99):
            seeds = arms[arm]
            for key, label in metrics[env]:
                vals = {sd: seeds[sd].get(key) for sd in sorted(seeds)}
                if all(v is None for v in vals.values()):
                    continue
                m, sd, n = mean_sd(list(vals.values()))
                per = " / ".join(f"s{x}:{fmt(v)}{'' if seeds[x]['finished'] else '*'}" for x, v in vals.items())
                dcol = "–"
                if arm != "grpo":
                    diffs = [vals[x] - base[x].get(key) for x in vals
                             if x in base and vals[x] is not None and base[x].get(key) is not None]
                    dm, lo, hi, dn = ci95(diffs)
                    if dm is not None:
                        dcol = f"{dm:+.1f} [{fmt(lo)}, {fmt(hi)}] (n={dn})"
                L.append(f"| {arm} | {label} | {fmt(m)} ± {fmt(sd)} | {n} | {per} | {dcol} |")
        L.append("")
    with open(os.path.join(res, "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"[collect] {len(runs)} result rows, {len(curves)} curve rows -> {res}/(runs.csv, curves.csv, summary.md)")


if __name__ == "__main__":
    main()
