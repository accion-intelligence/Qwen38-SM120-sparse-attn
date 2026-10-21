"""Time-to-first-token benchmark against a `vllm serve` instance launched by this script.

    python scripts/bench_ttft.py --model ... --prompts prompts.json --output ttft-fp8/ \
        --lengths 65536,131072,260096 --concurrencies 1,2 --repetitions 2 \
        --serve-args "--quantization fp8" --env SM120_SPARSE_KERNEL=sm120fp8 SM120_SPARSE_TAU=0.95

For each (length, concurrency) case the client fires `concurrency` streaming /v1/completions requests
(prompt token ids, greedy, ignore_eos) at once, records the time to the first generated token of each, keeps
streaming until every request has produced `--prefix-tokens` tokens (so first-token checks are possible), then
closes the connections and waits until the server is idle. One warmup run precedes `--repetitions` measured
runs. `results.json` has the same shape as summarize_ttft.py expects; `--reference` compares the checked
token prefixes against an earlier results.json (first-token agreement).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _engine import load_prompts  # noqa: E402


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def http_get(url, timeout=5):
    with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as r:
        return r.read().decode()


def wait_health(url, proc, timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f'vllm serve exited with {proc.returncode}')
        try:
            http_get(url + '/health')
            return
        except Exception:
            time.sleep(2)
    raise RuntimeError('server did not become healthy')


def running_requests(url):
    try:
        text = http_get(url + '/metrics')
    except Exception:
        return None
    total = 0.0
    for line in text.splitlines():
        if line.startswith(('vllm:num_requests_running', 'vllm:num_requests_waiting')):
            total += float(line.rsplit(' ', 1)[1])
    return total


def wait_idle(url, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if running_requests(url) == 0.0:
            return True
        time.sleep(1)
    return False


async def burst(url, model, requests, prefix_tokens, timeout):
    import httpx
    gate = asyncio.Event()
    done_prefix = set()
    all_done = asyncio.Event()
    rows = []
    async with httpx.AsyncClient(timeout=timeout, trust_env=False,
                                limits=httpx.Limits(max_connections=len(requests), max_keepalive_connections=len(requests))) as client:
        async def one(i, req):
            payload = {'model': model, 'prompt': req['prompt_token_ids'], 'max_tokens': req['output_tokens'], 'temperature': 0.0,
                       'top_p': 1.0, 'top_k': -1, 'ignore_eos': True, 'seed': 20261015, 'stream': True, 'stream_interval': 1,
                       'return_token_ids': True, 'skip_special_tokens': False, 'stream_options': {'include_usage': True}}
            msg = client.build_request('POST', url + '/v1/completions', content=json.dumps(payload).encode(),
                                       headers={'Content-Type': 'application/json'})
            row = {'request_id': req['request_id'], 'prompt_sha256': req['prompt_sha256'], 'ttft_ms': None, 'token_ids': []}
            await gate.wait()
            t0 = time.perf_counter_ns()
            resp = await client.send(msg, stream=True)
            try:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith('data: ') or line[6:] == '[DONE]':
                        continue
                    ev = json.loads(line[6:])
                    if ev.get('error'):
                        raise RuntimeError(str(ev['error']))
                    for ch in ev.get('choices', []):
                        toks = ch.get('token_ids') or []
                        if toks:
                            if row['ttft_ms'] is None:
                                row['ttft_ms'] = (time.perf_counter_ns() - t0) / 1e6
                            row['token_ids'].extend(toks)
                    if len(row['token_ids']) >= prefix_tokens:
                        done_prefix.add(i)
                        if len(done_prefix) == len(requests):
                            all_done.set()
                    if all_done.is_set():
                        break
            finally:
                await resp.aclose()
            row['checked_prefix'] = row.pop('token_ids')[:prefix_tokens]
            return row
        tasks = [asyncio.create_task(one(i, r)) for i, r in enumerate(requests)]
        gate.set()
        rows = await asyncio.gather(*tasks)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--served-model-name', default='model')
    ap.add_argument('--prompts', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True, help='new directory for results.json and server.log')
    ap.add_argument('--kinds', default='docs_needle')
    ap.add_argument('--lengths', default='65536,131072,260096')
    ap.add_argument('--concurrencies', default='1,2')
    ap.add_argument('--repetitions', type=int, default=2)
    ap.add_argument('--warmup', type=int, default=1)
    ap.add_argument('--prefix-tokens', type=int, default=1)
    ap.add_argument('--max-model-len', type=int, default=262144)
    ap.add_argument('--max-num-seqs', type=int, default=4)
    ap.add_argument('--max-num-batched-tokens', type=int, default=8192)
    ap.add_argument('--gpu-memory-utilization', type=float, default=0.9)
    ap.add_argument('--serve-args', default='', help='extra `vllm serve` arguments, e.g. "--quantization fp8"')
    ap.add_argument('--env', nargs='*', default=[], help='KEY=VALUE for the server (SM120_SPARSE_* ...)')
    ap.add_argument('--reference', type=Path, default=None, help='results.json of an earlier run for first-token checks')
    ap.add_argument('--timeout', type=float, default=3600)
    args = ap.parse_args()
    if args.output.exists():
        sys.exit(f'{args.output} exists; use a new directory')
    args.output.mkdir(parents=True)
    port = free_port()
    url = f'http://127.0.0.1:{port}'
    env = dict(os.environ)
    for kv in args.env:
        k, v = kv.split('=', 1)
        env[k] = v
    cmd = [sys.executable, '-m', 'vllm.entrypoints.cli.main', 'serve', args.model, '--served-model-name', args.served_model_name,
           '--host', '127.0.0.1', '--port', str(port), '--dtype', 'bfloat16', '--max-model-len', str(args.max_model_len),
           '--no-enable-prefix-caching', '--generation-config', 'vllm', '--seed', '20261015', '--max-num-seqs', str(args.max_num_seqs),
           '--max-num-batched-tokens', str(args.max_num_batched_tokens), '--gpu-memory-utilization', str(args.gpu_memory_utilization),
           '--enable-chunked-prefill', '--language-model-only'] + shlex.split(args.serve_args)
    log = open(args.output / 'server.log', 'w')
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    reference = {}
    if args.reference:
        for u in json.loads(args.reference.read_text())['uniform']:
            if not u['warmup']:
                for r in u['rows']:
                    reference.setdefault(r['request_id'], r['checked_prefix'])
    results = {'command': cmd, 'env': args.env, 'uniform': [], 'status': 'running'}
    try:
        wait_health(url, proc)
        prompts = load_prompts(args.prompts, args.kinds, args.lengths)
        for length in (int(x) for x in args.lengths.split(',')):
            for conc in (int(x) for x in args.concurrencies.split(',')):
                reqs = sorted([r for r in prompts if r['length'] == length], key=lambda r: r.get('index', 0))[:conc]
                if len(reqs) < conc:
                    print(f'skip length {length} x{conc}: only {len(reqs)} prompts', flush=True)
                    continue
                case = f"in{length}-out{reqs[0]['output_tokens']}-c{conc}"
                for rep in range(args.warmup + args.repetitions):
                    rows = asyncio.run(burst(url, args.served_model_name, reqs, args.prefix_tokens, args.timeout))
                    idle = wait_idle(url)
                    lat = [r['ttft_ms'] for r in rows]
                    passed, msg = True, 'ok'
                    for r in rows:
                        exp = reference.get(r['request_id'])
                        if exp is not None and exp[:args.prefix_tokens] != r['checked_prefix'][:len(exp[:args.prefix_tokens])]:
                            passed, msg = False, f"{r['request_id']}: first generated token differs from the reference"
                    rec = {'case': case, 'warmup': rep < args.warmup, 'repetition': rep - args.warmup + 1 if rep >= args.warmup else 0,
                           'metrics': {'ttft_mean_ms': statistics.fmean(lat), 'ttft_max_ms': max(lat)}, 'rows': rows,
                           'validation': {'passed': passed, 'message': msg}, 'server_idle': idle}
                    results['uniform'].append(rec)
                    print(f"{case} {'warmup' if rec['warmup'] else rec['repetition']} ttft_mean {rec['metrics']['ttft_mean_ms']:.1f} ms"
                          + ('' if passed else f'  CHECK FAILED: {msg}'), flush=True)
                    (args.output / 'results.json').write_text(json.dumps(results, indent=1))
        results['status'] = 'ok'
    finally:
        (args.output / 'results.json').write_text(json.dumps(results, indent=1))
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    print('done', args.output / 'results.json')


if __name__ == '__main__':
    main()
