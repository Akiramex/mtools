"""
ASR /request + /getResult benchmark test
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

ASR_PARAMS = {
    "lang_type": "zh-cmn-Hans-CN",
    "format": "wav",
    "output": "text",
    "sample_rate": "16000",
    "enable_modal_particle_filter": "true",
    "enable_inverse_text_normalization": "true",
    "gain": "1",
    "max_sentence_silence": "800",
    "decode_silence_segment": "false",
    "paragraph_condition": "200",
    "clusters": "0",
}


async def submit_task(
    client: httpx.AsyncClient, base_url: str, audio_path: str
) -> dict:
    with open(audio_path, "rb") as f:
        files = {"file": (Path(audio_path).name, f, "audio/wav")}
        resp = await client.post(f"{base_url}/request", data=ASR_PARAMS, files=files)
    resp.raise_for_status()
    return resp.json()


async def poll_result(
    client: httpx.AsyncClient, base_url: str, task_id: str, timeout: float = 6000
) -> dict:
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        resp = await client.get(f"{base_url}/getResult", params={"task_id": task_id})
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status", "")
        if status == "00000":
            return data
        if status == "20326" and data.get("message") != "":
            raise RuntimeError(f"Task failed: {data}")
        await asyncio.sleep(0.5)
    raise TimeoutError(f"Task {task_id} timed out after {timeout}s")


async def single_request(
    client: httpx.AsyncClient,
    base_url: str,
    audio_path: str,
    save_dir: str | None = None,
) -> dict:
    """Submit one task and wait for completion, return timing info."""
    t0 = time.perf_counter()
    submit_time = datetime.now().isoformat()
    submit_resp = await submit_task(client, base_url, audio_path)
    task_id = submit_resp["data"]["task_id"]
    t_submit = time.perf_counter() - t0

    t1 = time.perf_counter()
    result = await poll_result(client, base_url, task_id)
    t_poll = time.perf_counter() - t1
    t_total = time.perf_counter() - t0
    finish_time = datetime.now().isoformat()

    sentence_count = (
        len(result.get("data", [])) if isinstance(result.get("data"), list) else 0
    )

    return {
        "task_id": task_id,
        "t_submit": t_submit,
        "t_poll": t_poll,
        "t_total": t_total,
        "sentences": sentence_count,
        "submit_time": submit_time,
        "finish_time": finish_time,
        "result": result,
    }


async def run_sequential(
    base_url: str, audio_path: str, n: int, save_dir: str | None = None
):
    print(f"\n{'=' * 60}")
    print(f" Sequential test: {n} requests")
    print(f"{'=' * 60}")
    timings = []

    async with httpx.AsyncClient(timeout=60) as client:
        for i in range(n):
            print(f"  [{i + 1}/{n}] submitting...", end=" ", flush=True)
            info = await single_request(client, base_url, audio_path, save_dir)
            timings.append(info)
            print(f"done ({info['t_total']:.2f}s, {info['sentences']} sentences)")
            print(f"    submit: {info['submit_time']}  finish: {info['finish_time']}")

    print_stats("Sequential", timings)
    return timings


async def run_concurrent(
    base_url: str, audio_path: str, n: int, save_dir: str | None = None
):
    print(f"\n{'=' * 60}")
    print(f" Concurrent test: {n} parallel requests")
    print(f"{'=' * 60}")

    async with httpx.AsyncClient(timeout=60) as client:
        t0 = time.perf_counter()
        tasks = [
            single_request(client, base_url, audio_path, save_dir) for _ in range(n)
        ]
        timings = await asyncio.gather(*tasks)
        t_wall = time.perf_counter() - t0

    print_stats("Concurrent", list(timings))
    print(f"  Wall time (all {n}): {t_wall:.2f}s")
    for i, info in enumerate(timings):
        print(f"  [{i + 1}/{n}] submit: {info['submit_time']}  finish: {info['finish_time']}")
    return list(timings)


def print_stats(label: str, timings: list[dict]):
    if not timings:
        print("  No requests to summarize.")
        return
    totals = [t["t_total"] for t in timings]
    submits = [t["t_submit"] for t in timings]
    polls = [t["t_poll"] for t in timings]

    print(f"\n  --- {label} Stats ---")
    print(f"  {'':12s} {'avg':>8s} {'min':>8s} {'max':>8s} {'median':>8s}")
    for name, vals in [("submit", submits), ("poll", polls), ("total", totals)]:
        print(
            f"  {name:12s} {statistics.mean(vals):8.3f}s {min(vals):8.3f}s "
            f"{max(vals):8.3f}s {statistics.median(vals):8.3f}s"
        )
    if len(totals) > 1:
        print(f"  Throughput: {len(totals) / sum(totals):.2f} req/s")
    print(f"  Total sentences: {sum(t['sentences'] for t in timings)}")


async def main():
    parser = argparse.ArgumentParser(description="ASR benchmark")
    parser.add_argument(
        "--url", default="http://192.168.1.235:9090", help="Server base URL"
    )
    parser.add_argument(
        "--audio", default="../data/asr_example_zh.wav", help="Audio file path"
    )
    parser.add_argument("--seq", type=int, default=5, help="Sequential request count")
    parser.add_argument("--conc", type=int, default=5, help="Concurrent request count")
    parser.add_argument(
        "--save", default=None, help="Save getResult responses to this directory"
    )
    args = parser.parse_args()

    if not Path(args.audio).exists():
        print(f"Audio file not found: {args.audio}")
        sys.exit(1)

    print(f"Server:  {args.url}")
    print(f"Audio:   {args.audio} ({Path(args.audio).stat().st_size / 1024:.1f} KB)")
    if args.save:
        print(f"Save:    {args.save}")

    seq_timings = await run_sequential(args.url, args.audio, args.seq)
    conc_timings = await run_concurrent(args.url, args.audio, args.conc)

    if args.save:
        all_results = seq_timings + conc_timings
        out_file = Path(args.save)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(
            json.dumps(
                [{"task_id": t["task_id"],
                  "submit_time": t["submit_time"],
                  "finish_time": t["finish_time"],
                  "result": t["result"]}
                 for t in all_results],
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    asyncio.run(main())
