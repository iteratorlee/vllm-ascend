#!/usr/bin/env python3
"""Collect one cold-P window and one verified saturated-D diagnostic window."""

import argparse
import asyncio
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import httpx


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decode-window-s", type=float, default=1.0)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root / "bench"))
    # The existing benchmark supplies the deployed PD protocol and metrics.
    from bench_saturation import parse_metrics, request, sample_node, write_client_events

    args.output.mkdir(parents=True, exist_ok=True)
    p_url = "http://172.16.10.181:9081"
    d_urls = ["http://172.16.10.183:9900", "http://172.16.10.184:9900"]
    prompt = "请求编号 0000。" + (args.root / "bench/prompts/128k-context.txt").read_text()
    records = []
    origin = time.perf_counter()

    async with httpx.AsyncClient(timeout=3600, trust_env=False) as client:

        async def mark(action, url, payload=None):
            before = time.time()
            response = await client.post(url + action, json=payload)
            response.raise_for_status()
            records.append({"action": action, "url": url, "started_unix_s": before, "ended_unix_s": time.time()})
            (args.output / "control.json").write_text(json.dumps(records, indent=2) + "\n")
            return response

        for url in [p_url, *d_urls]:
            response = await client.get(url + "/health")
            response.raise_for_status()
        idle = await asyncio.gather(*[sample_node(client, url, origin) for url in d_urls])
        if any(node.get("running") != 0 or node.get("waiting") != 0 for node in idle):
            raise RuntimeError(f"D is not idle: {idle}")

        p_salt = "profile-cold-p-" + uuid.uuid4().hex
        p_body = {
            "model": "glm-5",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 1,
            "min_tokens": 1,
            "ignore_eos": True,
            "stream": False,
            "cache_salt": p_salt,
            "kv_transfer_params": {"do_remote_decode": False, "do_remote_prefill": False},
        }
        p_before = await client.get(p_url + "/metrics")
        p_before.raise_for_status()
        (args.output / "p-before.prom").write_text(p_before.text)
        try:
            await mark("/start_profile", p_url)
            response = await mark("/v1/chat/completions", p_url, p_body)
            p_result = response.json()
            assert p_result["usage"]["prompt_tokens"] == 131072, p_result.get("usage")
            assert p_result["usage"]["completion_tokens"] == 1, p_result.get("usage")
            (args.output / "p-request.json").write_text(
                json.dumps({"cache_salt": p_salt, "response": p_result}, ensure_ascii=False, indent=2) + "\n"
            )
        finally:
            await mark("/stop_profile", p_url)
        p_after = await client.get(p_url + "/metrics")
        p_after.raise_for_status()
        (args.output / "p-after.prom").write_text(p_after.text)
        before_metrics, after_metrics = parse_metrics(p_before.text), parse_metrics(p_after.text)
        p_prefix_delta = {key: after_metrics[key] - before_metrics[key] for key in ("prefix_hits", "prefix_queries")}
        if p_prefix_delta["prefix_hits"] != 0 or p_prefix_delta["prefix_queries"] <= 0:
            raise RuntimeError(f"P profiling did not verify cold-prefix computation: {p_prefix_delta}")

        workload = SimpleNamespace(
            prefill_url=p_url,
            decoder_urls=d_urls,
            output_tokens=1024,
            prompt_tokens=131072,
            collect_itl=True,
        )
        nonce = uuid.uuid4().hex
        warmup = await request(client, workload, prompt, -1, asyncio.Semaphore(1), origin, nonce, output_tokens=16)
        write_client_events(warmup, args.output)
        if warmup["error"] is not None:
            raise RuntimeError(warmup["error"])
        (args.output / "d-warmup.json").write_text(json.dumps(warmup, indent=2) + "\n")
        gate = asyncio.Semaphore(16)
        tasks = [asyncio.create_task(request(client, workload, prompt, i, gate, origin, nonce)) for i in range(16)]
        samples = []
        profile_attempted = False
        stop_errors = []
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                nodes = await asyncio.gather(*[sample_node(client, url, origin) for url in d_urls])
                samples.append({"unix_s": time.time(), "nodes": nodes})
                (args.output / "d-load.json").write_text(json.dumps(samples, indent=2) + "\n")
                if all(node.get("running") == 2 for node in nodes):
                    break
                if all(task.done() for task in tasks):
                    raise RuntimeError("D requests ended without observing 2+2 actual concurrency")
                await asyncio.sleep(0.2)
            else:
                raise TimeoutError("No measured 2+2 D concurrency within 180 seconds")
            profile_attempted = True
            await asyncio.gather(*[mark("/start_profile", url) for url in d_urls])
            nodes = await asyncio.gather(*[sample_node(client, url, origin) for url in d_urls])
            samples.append({"unix_s": time.time(), "stage": "profile-start", "nodes": nodes})
            if not all(node.get("running") == 2 for node in nodes):
                raise RuntimeError(f"D was not 2+2 when profiling started: {nodes}")
            await asyncio.sleep(args.decode_window_s)
            nodes = await asyncio.gather(*[sample_node(client, url, origin) for url in d_urls])
            samples.append({"unix_s": time.time(), "stage": "profile-end", "nodes": nodes})
            (args.output / "d-load.json").write_text(json.dumps(samples, indent=2) + "\n")
            if not all(node.get("running") == 2 for node in nodes):
                raise RuntimeError(f"D was not 2+2 when profiling ended: {nodes}")
        finally:
            if profile_attempted:
                stopped = await asyncio.gather(*[mark("/stop_profile", url) for url in d_urls], return_exceptions=True)
                stop_errors = [str(value) for value in stopped if isinstance(value, BaseException)]
            rows = await asyncio.gather(*tasks)
            for row in rows:
                write_client_events(row, args.output)
            (args.output / "d-requests.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
        if any(row["error"] is not None for row in rows):
            raise RuntimeError("Profiling workload failed; inspect d-requests.json")
        if stop_errors:
            raise RuntimeError(f"Profiler stop failed: {stop_errors}")
        (args.output / "manifest.json").write_text(
            json.dumps(
                {
                    "scope": "diagnostic profiling only, excluded from C03 performance comparison",
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "p": {"prompt_tokens": 131072, "output_tokens": 1, "new_prefix_salt": p_salt},
                    "p_prefix_counter_delta": p_prefix_delta,
                    "d": {"requests": 16, "client_concurrency": 16, "observed_running": [2, 2]},
                    "decode_window_s": args.decode_window_s,
                },
                indent=2,
            )
            + "\n"
        )
        print("Profiling workload completed", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
