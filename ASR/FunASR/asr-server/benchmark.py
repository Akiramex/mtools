"""
ASR Server 性能 & 并发测试脚本

用法: python benchmark.py [--url URL] [--audio AUDIO] [--concurrency N]

测试项:
  1. 单请求延迟 (ASR / SER / Speaker)
  2. 并发吞吐 (ASR 同步接口)
  3. 异步任务队列吞吐
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
import urllib.error
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple


# ── helpers ──────────────────────────────────────────────────────────────────

def post_file(url: str, file_path: str, fields: Optional[dict] = None) -> Tuple[int, dict, float]:
    """multipart/form-data POST, returns (status, json_body, elapsed_sec)."""
    boundary = uuid.uuid4().hex
    body = b""
    with open(file_path, "rb") as f:
        file_data = f.read()
    filename = file_path.replace("\\", "/").rsplit("/", 1)[-1]
    body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n\r\n".encode()
    body += file_data
    body += b"\r\n"
    if fields:
        for k, v in fields.items():
            body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
    body += f"--{boundary}--\r\n".encode()

    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
            return resp.status, data, time.perf_counter() - t0
    except urllib.error.HTTPError as e:
        data = json.loads(e.read()) if e.fp else {}
        return e.code, data, time.perf_counter() - t0


def get_json(url: str) -> Tuple[int, dict, float]:
    req = urllib.request.Request(url)
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
        return resp.status, data, time.perf_counter() - t0


def delete_json(url: str) -> Tuple[int, dict, float]:
    req = urllib.request.Request(url, method="DELETE")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
            return resp.status, data, time.perf_counter() - t0
    except urllib.error.HTTPError as e:
        data = json.loads(e.read()) if e.fp else {}
        return e.code, data, time.perf_counter() - t0


# ── tests ────────────────────────────────────────────────────────────────────

def test_health(base: str):
    print("\n[Health Check]")
    status, data, elapsed = get_json(f"{base}/health")
    print(f"  status={status}  elapsed={elapsed*1000:.1f}ms")
    print(f"  asr={data.get('asr_model_loaded')}  sv={data.get('sv_model_loaded')}  ser={data.get('ser_model_loaded')}")
    if not data.get("asr_model_loaded"):
        print("  FATAL: ASR model not loaded, abort.")
        sys.exit(1)
    return data


def test_single_asr(base: str, audio: str, rounds: int = 3):
    print(f"\n[ASR Single Request] x{rounds}")
    latencies = []
    for i in range(rounds):
        status, data, elapsed = post_file(f"{base}/asr/file", audio)
        latencies.append(elapsed)
        text = ""
        if data.get("data"):
            text = data["data"].get("text", "")[:50]
        print(f"  #{i+1}  status={status}  latency={elapsed*1000:.0f}ms  text=\"{text}...\"")
    _print_stats("ASR", latencies)


def test_single_asr_with_spk(base: str, audio: str, rounds: int = 3):
    print(f"\n[ASR + Speaker ID] x{rounds}")
    latencies = []
    for i in range(rounds):
        status, data, elapsed = post_file(
            f"{base}/asr/file", audio, fields={"identify_speakers": "true"}
        )
        latencies.append(elapsed)
        spk_count = 0
        if data.get("data"):
            spk_count = len(set(s.get("speaker") for s in data["data"].get("sentence_list", [])))
        print(f"  #{i+1}  status={status}  latency={elapsed*1000:.0f}ms  speakers={spk_count}")
    _print_stats("ASR+SPK", latencies)


def test_single_ser(base: str, audio: str, rounds: int = 3):
    print(f"\n[SER Single Request] x{rounds}")
    latencies = []
    for i in range(rounds):
        status, data, elapsed = post_file(f"{base}/ser/file", audio)
        latencies.append(elapsed)
        emotion = ""
        if data.get("data"):
            emotion = data["data"].get("emotion", "")
        print(f"  #{i+1}  status={status}  latency={elapsed*1000:.0f}ms  emotion=\"{emotion}\"")
    _print_stats("SER", latencies)


def test_speaker_register(base: str, audio: str, name: str = "bench_speaker"):
    print(f"\n[Speaker Register] name=\"{name}\"")
    # delete first if exists
    delete_json(f"{base}/speaker/{name}")
    status, data, elapsed = post_file(
        f"{base}/speaker/register", audio, fields={"name": name}
    )
    print(f"  status={status}  latency={elapsed*1000:.0f}ms  data={data}")
    return status == 200


def test_speaker_list(base: str):
    print("\n[Speaker List]")
    status, data, elapsed = get_json(f"{base}/speaker/list")
    speakers = data.get("data", [])
    print(f"  status={status}  latency={elapsed*1000:.0f}ms  count={len(speakers)}")


def test_speaker_delete(base: str, name: str = "bench_speaker"):
    print(f"\n[Speaker Delete] name=\"{name}\"")
    status, data, elapsed = delete_json(f"{base}/speaker/{name}")
    print(f"  status={status}  latency={elapsed*1000:.0f}ms")


def test_concurrent_asr(base: str, audio: str, concurrency: int, total: int):
    print(f"\n[ASR Concurrent] concurrency={concurrency}  total_requests={total}")
    latencies = []
    errors = 0

    def run_one(idx):
        try:
            status, data, elapsed = post_file(f"{base}/asr/file", audio)
            return status, elapsed
        except Exception as e:
            return -1, str(e)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(run_one, i) for i in range(total)]
        for future in as_completed(futures):
            status, val = future.result()
            if status == 200:
                latencies.append(val)
            else:
                errors += 1
                print(f"  ERROR: status={status}  val={val}")
    wall = time.perf_counter() - t0

    if latencies:
        _print_stats("ASR-Concurrent", latencies)
    rps = total / wall if wall > 0 else 0
    print(f"  wall_time={wall:.1f}s  throughput={rps:.2f} req/s  errors={errors}/{total}")


def test_concurrent_ser(base: str, audio: str, concurrency: int, total: int):
    print(f"\n[SER Concurrent] concurrency={concurrency}  total_requests={total}")

    latencies = []
    errors = 0

    def run_one(idx):
        try:
            status, data, elapsed = post_file(f"{base}/ser/file", audio)
            return status, elapsed
        except Exception as e:
            return -1, str(e)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(run_one, i) for i in range(total)]
        for future in as_completed(futures):
            status, val = future.result()
            if status == 200:
                latencies.append(val)
            else:
                errors += 1
                print(f"  ERROR: status={status}  val={val}")
    wall = time.perf_counter() - t0

    if latencies:
        _print_stats("SER-Concurrent", latencies)
    rps = total / wall if wall > 0 else 0
    print(f"  wall_time={wall:.1f}s  throughput={rps:.2f} req/s  errors={errors}/{total}")


def test_async_task_queue(base: str, audio: str, total: int, concurrency: int):
    print(f"\n[ASR Async Task Queue] total={total}  submit_concurrency={concurrency}")
    task_ids = []
    submit_latencies = []

    def submit_one(idx):
        status, data, elapsed = post_file(f"{base}/asr/task", audio)
        return status, data, elapsed

    # submit tasks concurrently
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(submit_one, i) for i in range(total)]
        for future in as_completed(futures):
            status, data, elapsed = future.result()
            submit_latencies.append(elapsed)
            if status == 200 and data.get("data"):
                task_ids.append(data["data"]["task_id"])
            else:
                print(f"  SUBMIT ERROR: status={status} data={data}")
    submit_wall = time.perf_counter() - t0
    print(f"  submitted={len(task_ids)}/{total}  wall={submit_wall:.1f}s")

    # poll until all done
    poll_start = time.perf_counter()
    completed = {}
    while len(completed) < len(task_ids):
        for tid in task_ids:
            if tid in completed:
                continue
            status, data, _ = get_json(f"{base}/asr/task/{tid}")
            if data.get("data") and data["data"].get("status") in ("completed", "failed"):
                completed[tid] = data["data"]
        if len(completed) < len(task_ids):
            time.sleep(0.5)
    poll_wall = time.perf_counter() - poll_start

    failed = sum(1 for v in completed.values() if v.get("status") == "failed")
    total_wall = time.perf_counter() - t0
    rps = len(task_ids) / total_wall if total_wall > 0 else 0

    print(f"  completed={len(completed)-failed}  failed={failed}  total_wall={total_wall:.1f}s")
    print(f"  effective_throughput={rps:.2f} tasks/s")
    _print_stats("Submit-latency", submit_latencies)


# ── util ─────────────────────────────────────────────────────────────────────

def _print_stats(label: str, latencies: List[float]):
    if not latencies:
        return
    ms = [l * 1000 for l in latencies]
    print(f"  [{label}] "
          f"min={min(ms):.0f}ms  avg={statistics.mean(ms):.0f}ms  "
          f"median={statistics.median(ms):.0f}ms  max={max(ms):.0f}ms  "
          f"p95={sorted(ms)[int(len(ms)*0.95)]:.0f}ms  n={len(ms)}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ASR Server Benchmark")
    parser.add_argument("--url", default="http://localhost:9090", help="Server base URL")
    parser.add_argument("--audio", default=None, help="Audio file path")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrent workers")
    parser.add_argument("--total", type=int, default=10, help="Total requests for concurrent tests")
    args = parser.parse_args()

    audio = args.audio
    if not audio:
        # try default locations
        candidates = [
            "../data/asr_example_zh.wav",
            "../../data/asr_example_zh.wav",
            "data/asr_example_zh.wav",
        ]
        for c in candidates:
            try:
                with open(c):
                    audio = c
                    break
            except FileNotFoundError:
                continue
    if not audio:
        print("ERROR: no audio file found. Use --audio to specify.")
        sys.exit(1)

    print(f"Audio: {audio}")
    print(f"Server: {args.url}")
    print(f"Concurrency: {args.concurrency}  Total requests: {args.total}")
    print("=" * 60)

    test_health(args.url)
    test_single_asr(args.url, audio, rounds=3)
    test_single_asr_with_spk(args.url, audio, rounds=3)
    test_single_ser(args.url, audio, rounds=3)

    test_speaker_register(args.url, audio)
    test_speaker_list(args.url)

    test_concurrent_asr(args.url, audio, concurrency=args.concurrency, total=args.total)
    test_concurrent_ser(args.url, audio, concurrency=args.concurrency, total=args.total)
    test_async_task_queue(args.url, audio, total=args.total, concurrency=args.concurrency)

    test_speaker_delete(args.url)

    print("\n" + "=" * 60)
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
