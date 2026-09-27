"""Multi-threaded judging of SurveillanceVQA predictions (perry: 评分尽量多线程并行).

Uses SurveillanceVQA's OWN four judge prompts (verbatim from
surveillancevqa_judge_prompts.json) and combines them with an EQUAL-WEIGHT mean
(perry 2026-09-27; matches the official paper's unweighted Avg). Writes scores back,
keeping the raw judge reply for later auditing. Resumable: records already judged
(and parseable) are skipped.
"""
from __future__ import annotations
import argparse, json, os, re, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = json.loads((ROOT/"benchmark/st_rare/surveillancevqa_judge_prompts.json").read_text())
DIMS = ["CI", "DO", "CU", "TU"]


def judge_once(question, ref, pred, dim, api_key, max_retries=3):
    from zhipuai import ZhipuAI
    tx = PROMPTS[dim]["prompt_texts"]
    msg = (f"{tx[0]}\n" + "\n".join(tx[1:]) +
           f"\n\nQuestion:\n{question}\n\nCorrect answer:\n{ref}\n\nPredicted answer:\n{pred}\n\n"
           f"Provide ONLY the integer 0-5.")
    for _ in range(max_retries):
        try:
            r = ZhipuAI(api_key=api_key).chat.completions.create(
                model="glm-4-flash", messages=[{"role": "user", "content": msg}], temperature=0.0)
            raw = (r.choices[0].message.content or "").strip()
            m = re.search(r"\b([0-5])\b", raw)
            return (int(m.group(1)) if m else None), raw
        except Exception as e:                                     # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
    return None, last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dims", nargs="+", default=DIMS)
    ap.add_argument("--api-key", default=os.environ.get("ZHIPUAI_API_KEY"),
                    help="ZhipuAI API key (or set the ZHIPUAI_API_KEY environment variable)")
    args = ap.parse_args()

    key = args.api_key
    if not key:
        raise SystemExit("No ZhipuAI API key: pass --api-key or set ZHIPUAI_API_KEY")
    recs = [json.loads(l) for l in args.preds.read_text().splitlines() if l.strip()]
    done = {}
    if args.output.exists():
        for l in args.output.read_text().splitlines():
            if l.strip():
                try:
                    x = json.loads(l); done[(x["stem"], x["family"], x["method"])] = x
                except json.JSONDecodeError:
                    pass
    print(f"  待评 {len(recs)} 条，已有 {len(done)} 条（跳过）", flush=True)

    lock = Lock(); handle = args.output.open("a", encoding="utf-8"); n = [0]
    def work(rec):
        scores = {d: judge_once(rec["question"], rec["reference"], rec["prediction"], d, key)[0]
                  for d in args.dims}
        raw = {d: judge_once(rec["question"], rec["reference"], rec["prediction"], d, key)[1]
               for d in args.dims} if any(v is None for v in scores.values()) else {}
        out = dict(rec); out["judge"] = scores; out["judge_raw"] = raw
        out["avg"] = (None if any(v is None for v in scores.values())
                      else sum(scores.values()) / len(args.dims))
        return out

    todo = [r for r in recs if (r["stem"], r["family"], r["method"]) not in done]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, r): r for r in todo}
        for f in as_completed(futs):
            o = f.result()
            with lock:
                handle.write(json.dumps(o, ensure_ascii=False) + "\n"); handle.flush()
                n[0] += 1
                if n[0] % 200 == 0:
                    print(f"    已评 {n[0]}/{len(todo)}", flush=True)
    handle.close(); print(f"  完成：本次评判 {n[0]} 条 -> {args.output}")


if __name__ == "__main__":
    main()
