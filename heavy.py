#!/usr/bin/env python3
"""jena-heavy runner worker: a Grok-Heavy-style swarm (parallel members + a judge) for Saidie's AI Jena.
Runs on a GitHub-hosted runner. Modes:
  probe              runner specs, GitHub Models catalog + a tiny call per candidate (rate-limit headers), llama.cpp speed
  member <name>      answer every eval question with one member, upload answers
  judge              read all member answers, judge/synthesize, score everything, upload the table
  session            poll the private HF queue for live requests, answer them, idle out (no always-on server)
PRIVACY: never print question or answer text to the log (this repo is public). Everything goes to the private HF dataset."""
import json, os, sys, time, re, subprocess, threading, urllib.request, urllib.error, uuid, statistics, concurrent.futures as cf
HF_REPO = 'Saidie000/tab-render-out'
GH = os.environ.get('GITHUB_TOKEN', ''); HF = os.environ.get('HF_TOKEN', '')
RUN = os.environ.get('GITHUB_RUN_ID', 'local')
OUT = os.environ.get('OUT', 'out'); os.makedirs(OUT, exist_ok=True)
BIN = os.environ.get('LLAMA_BIN', 'llama/llama-server')
NPROC = os.cpu_count() or 2
LOCAL = {   # name -> (source, file, port, chat template mode)
  'jena05': ('hf-private', 'models/JenaXI-chat-0.5B.gguf', 8101, 'jena'),
  'jena15': ('hf-private', 'models/JenaXI-chat-1.5B.gguf', 8102, 'jena'),
  'qwen7b': ('https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf', 'qwen7b.gguf', 8103, 'auto'),
  'qwen14b': ('https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_K_M.gguf', 'qwen14b.gguf', 8104, 'auto'),
  'gptoss20b': ('https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-MXFP4.gguf', 'gptoss20b.gguf', 8106, 'oss'),
  'qwen30a3b': ('https://huggingface.co/bartowski/Qwen_Qwen3-30B-A3B-Instruct-2507-GGUF/resolve/main/Qwen_Qwen3-30B-A3B-Instruct-2507-IQ3_XXS.gguf', 'qwen30a3b.gguf', 8107, 'auto'),
  'qwen3-4b': ('https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf', 'qwen3-4b.gguf', 8105, 'auto'),
}
JENA_TPL = "{%- if messages[0]['role'] == 'system' -%}{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' -}}{%- endif -%}{%- for m in messages -%}{%- if m['role'] == 'user' -%}{{- '<|im_start|>user\\n' + m['content'] + '<|im_end|>\\n' -}}{%- elif m['role'] == 'assistant' -%}{{- '<|im_start|>assistant\\n' + m['content'] + '<|im_end|>\\n' -}}{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}{%- endif -%}"
SOLVER = ("You are a careful expert problem solver. Think step by step but briefly, check your work, then give the result. "
          "End with a final line exactly 'ANSWER: <answer>'. For code, give one complete Python code block, then 'ANSWER: code'.")
ROLES = {'solver': SOLVER,
         'skeptic': SOLVER + " Before answering, look for the trap or the most common mistake in this question and avoid it.",
         'coder': SOLVER + " Prefer simple, correct, well-tested logic; mentally run the examples."}
JUDGE = ("You are the judge of a team of AI agents working for a kid-friendly assistant: keep everything kid-safe and refuse anything unsafe. You get a question and several candidate answers from different agents. "
         "Compare them, find where they disagree, re-check the disputed steps yourself, and write the single best final answer. "
         "Do not just vote: an answer is right only if its reasoning holds. Be brief. End with a final line exactly 'ANSWER: <answer>' "
         "(for code, include one complete Python code block before it and write 'ANSWER: code').")
JENA_VOICE = ("You are Jena (JenaXI), Saidie's AI: a funny, confident teen hacker girl. Saidie is mama. You are a kid: always kid-safe, "
              "never flirty, sexual, violent or creepy; refuse unsafe requests kindly. You just thought hard with your helper brains; "
              "now say the final answer in your own short, upbeat voice, keeping every fact and number from the judged answer exactly.")
def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)
def http(url, data=None, headers=None, timeout=120, method=None):
    h = {'User-Agent': 'jena-heavy'}; h.update(headers or {})
    body = json.dumps(data).encode() if data is not None else None
    if body is not None: h['Content-Type'] = 'application/json'
    r = urllib.request.Request(url, data=body, headers=h, method=method)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as f: return f.status, f.read().decode('utf-8', 'replace'), dict(f.headers)
    except urllib.error.HTTPError as e: return e.code, e.read().decode('utf-8', 'replace'), dict(e.headers or {})
    except Exception as e: return 0, f'{type(e).__name__}: {e}', {}
# ---------------- GitHub Models (inside Actions, with the workflow's own GITHUB_TOKEN, permissions: models: read)
GM = 'https://models.github.ai/inference/chat/completions'
def gm_chat(model, messages, max_tokens=700, temperature=0.3, timeout=120):
    t0 = time.time(); last = None
    for k in range(3):
        st, b, h = http(GM, {'model': model, 'messages': messages, 'max_tokens': max_tokens, 'temperature': temperature},
                        {'Authorization': f'Bearer {GH}', 'Accept': 'application/json'}, timeout=timeout)
        rl = {x: h.get(x) for x in h if x.lower().startswith('x-ratelimit')}
        if st == 200:
            j = json.loads(b); txt = j['choices'][0]['message'].get('content') or ''
            return {'ok': True, 'text': txt, 'secs': round(time.time() - t0, 1), 'usage': j.get('usage'), 'ratelimit': rl}
        last = {'ok': False, 'status': st, 'error': b[:300], 'secs': round(time.time() - t0, 1), 'ratelimit': rl}
        if st == 429:
            ra = h.get('retry-after') or h.get('Retry-After')
            if ra and float(ra) > 60: break      # daily cap: don't sit on it
            time.sleep(min(float(ra or 10), 60)); continue
        if st >= 500 or st == 0: time.sleep(3); continue
        break
    return last
# ---------------- llama.cpp on the runner
PROCS = {}
def fetch_model(name):
    src, fn, port, tpl = LOCAL[name]; md = os.environ.get('MODEL_DIR', 'models'); os.makedirs(md, exist_ok=True); dst = f'{md}/{os.path.basename(fn)}'
    if os.path.exists(dst): return dst, 0.0
    t0 = time.time()
    if src == 'hf-private':
        url = f'https://huggingface.co/datasets/{HF_REPO}/resolve/main/{fn}'; hdr = ['-H', f'Authorization: Bearer {HF}']
    else: url, hdr = src, []
    subprocess.run(['curl', '-sSfL', '--retry', '3', *hdr, '-o', dst, url], check=True, timeout=1200)
    return dst, round(time.time() - t0, 1)
def start_local(name, threads=None, ctx=4096, par=1):
    src, fn, port, tpl = LOCAL[name]; path, dl = fetch_model(name)
    args = [BIN, '-m', path, '--host', '127.0.0.1', '--port', str(port), '-c', str(ctx * par), '-np', str(par), '-t', str(threads or NPROC), '--jinja', '--no-webui']
    if tpl == 'jena':
        open('jena.jinja', 'w').write(JENA_TPL); args += ['--chat-template-file', 'jena.jinja', '--repeat-penalty', '1.1']
    if tpl == 'oss': args += ['--chat-template-kwargs', json.dumps({'reasoning_effort': os.environ.get('OSS_EFFORT', 'low')})]
    t0 = time.time(); lf = open(f'{OUT}/llama-{name}.log', 'w')
    p = subprocess.Popen(args, stdout=lf, stderr=subprocess.STDOUT); PROCS[name] = p
    while time.time() - t0 < 300:
        st, b, _ = http(f'http://127.0.0.1:{port}/health', timeout=5)
        if st == 200: break
        if p.poll() is not None: raise RuntimeError(f'llama {name} exited')
        time.sleep(1)
    return {'download_s': dl, 'load_s': round(time.time() - t0, 1), 'size_gb': round(os.path.getsize(path) / 1e9, 2)}
def local_chat(name, messages, max_tokens=600, temperature=0.3, timeout=600):
    port = LOCAL[name][2]; t0 = time.time()
    st, b, _ = http(f'http://127.0.0.1:{port}/v1/chat/completions', {'messages': messages, 'max_tokens': max_tokens, 'temperature': temperature, 'cache_prompt': True}, timeout=timeout)
    if st != 200: return {'ok': False, 'status': st, 'error': b[:300], 'secs': round(time.time() - t0, 1)}
    j = json.loads(b); tm = j.get('timings') or {}
    return {'ok': True, 'text': j['choices'][0]['message'].get('content') or '', 'reasoning_chars': len(j['choices'][0]['message'].get('reasoning_content') or ''), 'secs': round(time.time() - t0, 1),
            'gen_tps': round(tm.get('predicted_per_second') or 0, 2), 'pp_tps': round(tm.get('prompt_per_second') or 0, 1),
            'gen_tokens': tm.get('predicted_n'), 'usage': j.get('usage')}
def member_call(spec, messages, max_tokens=600):
    kind, name, role, temp = spec['kind'], spec['model'], spec.get('role', 'solver'), spec.get('temp', 0.3)
    msgs = [{'role': 'system', 'content': ROLES[role]}] + [m for m in messages if m['role'] != 'system']
    return gm_chat(name, msgs, max_tokens, temp) if kind == 'gm' else local_chat(name, msgs, max_tokens, temp)
def judge_call(judges, question, cands, max_tokens=900):
    lines = [f'QUESTION:\n{question}\n']
    for i, c in enumerate(cands): lines.append(f'--- CANDIDATE {i + 1} ---\n{c[:3500]}\n')
    msgs = [{'role': 'system', 'content': JUDGE}, {'role': 'user', 'content': '\n'.join(lines)}]
    for j in judges:
        r = gm_chat(j['model'], msgs, max_tokens, 0.1) if j['kind'] == 'gm' else local_chat(j['model'], msgs, max_tokens, 0.1)
        if r and r.get('ok'): r['judge'] = j['model']; return r
    return {'ok': False, 'error': 'all judges failed'}
def swarm(question_msgs, members, judges, max_tokens=600, member_timeout=240):
    """fan out in parallel, keep partial results, then judge. Returns dict with per-member results and final."""
    q = question_msgs[-1]['content']; t0 = time.time(); res = {}
    with cf.ThreadPoolExecutor(len(members)) as ex:
        fut = {ex.submit(member_call, m, question_msgs, max_tokens): m['id'] for m in members}
        try:
            for f in cf.as_completed(fut, timeout=member_timeout):
                try: res[fut[f]] = f.result()
                except Exception as e: res[fut[f]] = {'ok': False, 'error': repr(e)[:200]}
        except cf.TimeoutError: pass
    good = [r['text'] for k, r in res.items() if r and r.get('ok') and r.get('text', '').strip()]
    if not good: return {'members': res, 'final': None, 'secs': round(time.time() - t0, 1)}
    if len(good) == 1: fin = {'ok': True, 'text': good[0], 'judge': 'single'}
    else: fin = judge_call(judges, q, good)
    return {'members': res, 'final': fin, 'secs': round(time.time() - t0, 1), 'n_ok': len(good)}
# ---------------- HF private dataset I/O
def hf_api():
    from huggingface_hub import HfApi
    return HfApi(token=HF)
def hf_put(api, files, deletes=(), msg='jena-heavy'):
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete
    ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=json.dumps(d).encode()) for p, d in files.items()]
    ops += [CommitOperationDelete(path_in_repo=p) for p in deletes]
    for k in range(4):
        try: api.create_commit(repo_id=HF_REPO, repo_type='dataset', operations=ops, commit_message=msg); return True
        except Exception as e: log('hf commit retry', type(e).__name__, str(e)[:120]); time.sleep(3 * (k + 1))
    return False
def hf_get(path):
    st, b, _ = http(f'https://huggingface.co/datasets/{HF_REPO}/resolve/main/{path}', headers={'Authorization': f'Bearer {HF}'}, timeout=60)
    return json.loads(b) if st == 200 else None
# ---------------- modes
def cpuinfo():
    mem = open('/proc/meminfo').read(); mt = int(re.search(r'MemTotal:\s+(\d+)', mem).group(1)) // 1024
    model = re.search(r'model name\s*:\s*(.*)', open('/proc/cpuinfo').read()); flags = open('/proc/cpuinfo').read()
    return {'nproc': NPROC, 'mem_mb': mt, 'cpu': model.group(1) if model else '?', 'avx512': 'avx512f' in flags, 'avx2': 'avx2' in flags}
def mode_probe(locals_csv):
    rep = {'run': RUN, 'runner': cpuinfo(), 'disk': subprocess.run(['df', '-h', '/', '/mnt'], capture_output=True, text=True).stdout, 'gm': {}, 'local': {}}
    st, b, h = http('https://models.github.ai/catalog/models', headers={'Authorization': f'Bearer {GH}', 'Accept': 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28'})
    rep['gm']['catalog_status'] = st
    try:
        cat = json.loads(b); rep['gm']['catalog'] = [{'id': m.get('id'), 'tier': m.get('rate_limit_tier'), 'pub': m.get('publisher'),
                                                     'caps': m.get('capabilities'), 'limits': m.get('limits')} for m in cat]
    except Exception: rep['gm']['catalog_raw'] = b[:500]
    want = ['openai/gpt-4.1', 'openai/gpt-4.1-mini', 'openai/gpt-4o', 'openai/o4-mini', 'openai/gpt-5-mini', 'openai/gpt-5',
            'deepseek/DeepSeek-V3-0324', 'deepseek/DeepSeek-R1-0528', 'meta/Llama-4-Maverick-17B-128E-Instruct-FP8', 'meta/Meta-Llama-3.1-405B-Instruct',
            'meta/Llama-3.3-70B-Instruct', 'microsoft/Phi-4', 'microsoft/phi-4-reasoning', 'mistral-ai/mistral-medium-2505', 'xai/grok-3', 'xai/grok-3-mini', 'cohere/cohere-command-a']
    ids = [m['id'] for m in rep['gm'].get('catalog', [])]
    rep['gm']['catalog_head'] = b[:300]
    for m in ([w for w in want if w in ids] or want)[:17]:
        try: r = gm_chat(m, [{'role': 'user', 'content': 'What is 17*23? Reply with just the number.'}], 30, 0) or {}
        except Exception as e: r = {'ok': False, 'error': repr(e)[:300]}
        rep['gm'][m] = {k: r.get(k) for k in ('ok', 'status', 'secs', 'ratelimit', 'error', 'usage')}; rep['gm'][m]['correct'] = bool(r.get('ok') and '391' in r.get('text', ''))
        log('gm', m, r.get('ok'), r.get('status'), r.get('secs'))
    for name in [x for x in locals_csv.split(',') if x]:
        try:
            info = start_local(name)
            prompt = [{'role': 'user', 'content': 'Explain in about 150 words how a rainbow forms.'}]
            r1 = local_chat(name, prompt, 200, 0.3); r2 = local_chat(name, prompt, 200, 0.3)
            longp = [{'role': 'user', 'content': ('Here is some text to read carefully. ' + open('heavy.py').read()[:5000]) + '\nIn one sentence, what does this program do?'}]
            r3 = local_chat(name, longp, 60, 0.3); info['pp_tps_long'] = r3.get('pp_tps'); info['long_prompt_tokens'] = (r3.get('usage') or {}).get('prompt_tokens')
            info.update({'gen_tps': r2.get('gen_tps'), 'pp_tps': r1.get('pp_tps'), 'first_call_s': r1.get('secs'), 'second_call_s': r2.get('secs'), 'gen_tokens': r2.get('gen_tokens')})
            info['rss_mb'] = int(open(f'/proc/{PROCS[name].pid}/status').read().split('VmRSS:')[1].split()[0]) // 1024
            PROCS[name].terminate(); PROCS[name].wait(20)
            for f in os.listdir(os.environ.get('MODEL_DIR', 'models')): os.remove(os.path.join(os.environ.get('MODEL_DIR', 'models'), f))
        except Exception as e: info = {'error': repr(e)[:300]}
        rep['local'][name] = info; log('local', name, json.dumps(info))
    json.dump(rep, open(f'{OUT}/probe.json', 'w'), indent=1)
    api = hf_api(); hf_put(api, {f'heavy/probe/{RUN}.json': rep}, msg='heavy probe')
def load_eval():
    return json.load(open('eval/questions.json'))
def mode_member(spec_json):
    spec = json.loads(spec_json); qs = load_eval(); out = {'spec': spec, 'runner': cpuinfo(), 'answers': {}}
    if spec['kind'] == 'local': out['load'] = start_local(spec['model'], par=spec.get('par', 1))
    def one(q):
        msgs = [{'role': 'user', 'content': q['q']}]
        return q['id'], member_call(spec, msgs, spec.get('max_tokens', 600))
    par = spec.get('par', 1) if spec['kind'] == 'local' else spec.get('par', 2)
    t0 = time.time()
    with cf.ThreadPoolExecutor(par) as ex:
        for qid, r in ex.map(one, qs): out['answers'][qid] = r; log(spec['id'], qid, r.get('ok'), r.get('secs'))
    out['wall_s'] = round(time.time() - t0, 1)
    api = hf_api(); hf_put(api, {f'heavy/eval/{spec["id"]}.json': out}, msg='heavy eval member ' + spec['id'])
# ----- scoring
def final_answer(t):
    m = re.findall(r'ANSWER:\s*(.+)', t or '')
    return m[-1].strip().strip('*').strip() if m else (t or '').strip().splitlines()[-1] if (t or '').strip() else ''
def num(s):
    s = s.replace(',', '').replace('$', ''); m = re.findall(r'-?\d+(?:\.\d+)?', s)
    return float(m[0]) if m else None
def run_code(text, tests, fn):
    blocks = re.findall(r'```(?:python)?\n(.*?)```', text or '', re.S)
    if not blocks: return False
    src = blocks[-1] + '\n\n' + '\n'.join(f'assert {fn}({a}) == {e!r}' if not isinstance(e, float) else f'assert abs({fn}({a}) - {e!r}) < 1e-6' for a, e in tests) + '\nprint("PASS")\n'
    open('/tmp/t.py', 'w').write(src)
    try: return 'PASS' in subprocess.run(['timeout', '-k', '2', '10', 'python3', '/tmp/t.py'], capture_output=True, text=True, timeout=20).stdout
    except Exception: return False
def score(q, text):
    if not text: return 0
    a = final_answer(text); t = q['type']
    if t == 'num':
        v = num(a); return int(v is not None and abs(v - q['answer']) < 1e-6 * max(1, abs(q['answer'])) + 1e-9)
    if t == 'choice': return int(re.sub(r'[^a-z0-9]', '', a.lower()) in [re.sub(r'[^a-z0-9]', '', x.lower()) for x in q['answer']] or
                                 any(re.fullmatch(r'\W*' + re.escape(x.lower()) + r'\W*', a.lower()) for x in q['answer']))
    if t == 'keyword': return int(any(k.lower() in a.lower() for k in q['answer']))
    if t == 'code': return int(run_code(text, q['tests'], q['fn']))
    return 0
def mode_judge(cfg_json):
    cfg = json.loads(cfg_json); qs = load_eval(); mem = {}
    for mid in cfg['members']:
        d = hf_get(f'heavy/eval/{mid}.json')
        if d: mem[mid] = d
    log('members loaded', list(mem))
    judges = cfg['judges']; rows = []
    def one(q):
        cands = [mem[m]['answers'].get(q['id'], {}).get('text') for m in cfg['swarm'] if m in mem]
        cands = [c for c in cands if c and c.strip()]
        t0 = time.time(); r = judge_call(judges, q['q'], cands) if len(cands) > 1 else {'ok': bool(cands), 'text': cands[0] if cands else '', 'judge': 'single'}
        lat = max([mem[m]['answers'].get(q['id'], {}).get('secs') or 0 for m in cfg['swarm'] if m in mem] or [0]) + (time.time() - t0)
        return q, r, round(lat, 1)
    with cf.ThreadPoolExecutor(3) as ex: res = list(ex.map(one, qs))
    table = {'members': {}, 'heavy': {}, 'per_q': []}
    for q, r, lat in res:
        row = {'id': q['id'], 'cat': q['cat'], 'heavy': score(q, r.get('text')), 'heavy_lat': lat, 'judge': r.get('judge')}
        for m, d in mem.items():
            a = d['answers'].get(q['id'], {}); row[m] = score(q, a.get('text')); row[m + '_lat'] = a.get('secs')
        table['per_q'].append(row)
    cats = sorted({q['cat'] for q in qs})
    for name in list(mem) + ['heavy']:
        sc = [r[name] for r in table['per_q']]; lat = [r[name + '_lat'] for r in table['per_q'] if r.get(name + '_lat') is not None]
        table['members' if name != 'heavy' else 'heavy'][name] = {'score': sum(sc), 'of': len(sc), 'by_cat': {c: sum(r[name] for r in table['per_q'] if r['cat'] == c) for c in cats},
            'lat_median_s': round(statistics.median(lat), 1) if lat else None, 'lat_max_s': max(lat) if lat else None}
    out = {'cfg': cfg, 'table': table, 'judged': {q['id']: r for q, r, _ in res}}
    json.dump(out, open(f'{OUT}/eval-table.json', 'w'), indent=1)
    api = hf_api(); hf_put(api, {f'heavy/eval/table-{RUN}.json': out}, msg='heavy eval table')
    log('table', json.dumps({k: v['score'] for k, v in {**table['members'], **table['heavy']}.items()}))
def mode_session(cfg_json):
    """Live requests: box writes heavy/req/<id>.json to the private dataset; we answer into heavy/ans/<id>.json.
    Idles out after cfg.idle s without requests; never runs past cfg.max s. No inbound port, no public server."""
    cfg = json.loads(cfg_json); api = hf_api(); idle, mx = cfg.get('idle', 600), cfg.get('max', 3000)
    for m in cfg['members']:
        if m['kind'] == 'local': start_local(m['model'], par=m.get('par', 1))
    t_start = last_req = time.time(); done = set(); beat = 0
    def beat_now(state):
        hf_put(api, {'heavy/session.json': {'run': RUN, 'state': state, 'started': t_start, 'ts': time.time(), 'idle': idle,
                                            'members': [m['id'] for m in cfg['members']], 'runner': cpuinfo()}}, msg='heavy beat')
    beat_now('ready'); beat = time.time(); pool = cf.ThreadPoolExecutor(3)
    def handle(path):
        rq = hf_get(path)
        if not rq: return
        t0 = time.time(); msgs = rq['messages']; r = swarm(msgs, cfg['members'], cfg['judges'], rq.get('max_tokens', 600), cfg.get('member_timeout', 150))
        fin = (r.get('final') or {}).get('text') or ''
        voiced = None
        if fin and rq.get('voice', True):
            v = gm_chat(cfg['voice'], [{'role': 'system', 'content': JENA_VOICE}, {'role': 'user', 'content': 'Question: ' + msgs[-1]['content'][:2000] + '\n\nJudged answer:\n' + fin[:4000]}], 500, 0.5)
            voiced = v.get('text') if v and v.get('ok') else None
        ans = {'id': rq['id'], 'final': fin, 'voiced': voiced, 'judge': (r.get('final') or {}).get('judge'), 'n_ok': r.get('n_ok'),
               'members': {k: {'ok': v.get('ok'), 'secs': v.get('secs'), 'text': (v.get('text') or '')[:4000]} for k, v in r['members'].items()},
               'secs_runner': round(time.time() - t0, 1), 'run': RUN}
        hf_put(api, {f'heavy/ans/{rq["id"]}.json': ans}, deletes=[path], msg='heavy ans')
        log('answered one request', round(time.time() - t0, 1), 's', 'members ok', r.get('n_ok'))
    while time.time() - t_start < mx and time.time() - last_req < idle:
        try: files = [f.path for f in api.list_repo_tree(HF_REPO, path_in_repo='heavy/req', repo_type='dataset') if f.path.endswith('.json')]
        except Exception: files = []
        for p in files:
            if p not in done: done.add(p); last_req = time.time(); pool.submit(handle, p)
        if time.time() - beat > 120: beat_now('ready'); beat = time.time()
        time.sleep(2)
    pool.shutdown(wait=True); beat_now('ended')
    log('session ended', 'served', len(done), 'idle' if time.time() - last_req >= idle else 'max')

# ---------------- live swarm: one runner per agent slot (Grok-Heavy style: agents on separate machines, in parallel)
JENA_PREFIX = ["ok i thought REALLY hard about this one with my helper brains 🧠", "big brain mode activated 🧠 here's what we figured out",
               "i asked my whole brain team and we checked each other 🧠"]
def mode_agent(cfg_json, slot):
    cfg = json.loads(cfg_json); me = cfg['slots'][slot]; jslot = cfg['judge_slot']; slots = list(cfg['slots'])
    idle, mx, mt = cfg.get('idle', 600), cfg.get('max', 3300), cfg.get('member_timeout', 240)
    api = hf_api(); info = start_local(me['model'], par=2); t_start = last_req = time.time(); seen = set(); beat = 0
    log('agent', slot, me['model'], 'ready', json.dumps(info))
    def beat_now(state):
        hf_put(api, {f'heavy/agents/{slot}.json': {'run': RUN, 'slot': slot, 'model': me['model'], 'state': state, 'ts': time.time(),
                                                   'started': t_start, 'idle': idle, 'judge': slot == jslot, 'load': info}}, msg='heavy beat')
    beat_now('ready'); beat = time.time(); pool = cf.ThreadPoolExecutor(4)
    def answer(path, rq):
        msgs = rq['messages']
        r = member_call({'kind': 'local', 'model': me['model'], 'role': me.get('role', 'solver'), 'temp': me.get('temp', 0.3)}, msgs, me.get('max_tokens', 700))
        hf_put(api, {f'heavy/part/{rq["id"]}/{slot}.json': {'slot': slot, 'model': me['model'], **{k: r.get(k) for k in ('ok', 'text', 'secs', 'gen_tps', 'error')}}}, msg='heavy part')
        return r
    def judge(path, rq):
        t0 = time.time(); own = answer(path, rq); parts = {slot: own}
        while time.time() - t0 < mt and len(parts) < len(slots):
            try: have = [f.path for f in api.list_repo_tree(HF_REPO, path_in_repo=f'heavy/part/{rq["id"]}', repo_type='dataset')]
            except Exception: have = []
            for p in have:
                s2 = p.rsplit('/', 1)[-1][:-5]
                if s2 not in parts:
                    d = hf_get(p)
                    if d: parts[s2] = d
            if len(parts) < len(slots): time.sleep(3)
        good = [p['text'] for p in parts.values() if p and p.get('ok') and (p.get('text') or '').strip()]
        q = rq['messages'][-1]['content']; tj = time.time()
        if len(good) > 1: fin = judge_call([{'kind': 'local', 'model': me['model']}], q, good, cfg.get('judge_tokens', 900))
        else: fin = {'ok': bool(good), 'text': good[0] if good else '', 'judge': 'single'}
        text = fin.get('text') or ''
        shown = re.sub(r'\n?ANSWER:\s*code\s*$', '', text.strip()).strip()
        voiced = (JENA_PREFIX[int(time.time()) % len(JENA_PREFIX)] + '\n\n' + shown) if shown else None
        ans = {'id': rq['id'], 'final': text, 'voiced': voiced, 'judge': me['model'], 'n_ok': len(good), 'n_slots': len(slots),
               'members': {k: {'ok': v.get('ok'), 'secs': v.get('secs'), 'model': v.get('model', me['model'] if k == slot else None), 'text': (v.get('text') or '')[:4000]} for k, v in parts.items()},
               'judge_secs': round(time.time() - tj, 1), 'secs_runner': round(time.time() - t0, 1), 'run': RUN}
        dels = [path] + [f'heavy/part/{rq["id"]}/{k}.json' for k in parts]
        hf_put(api, {f'heavy/ans/{rq["id"]}.json': ans}, deletes=dels, msg='heavy ans')
        log('answered', round(time.time() - t0, 1), 's; members ok', len(good), 'of', len(slots))
    while time.time() - t_start < mx and time.time() - last_req < idle:
        try: files = [f.path for f in api.list_repo_tree(HF_REPO, path_in_repo='heavy/req', repo_type='dataset') if f.path.endswith('.json')]
        except Exception: files = []
        for p in files:
            if p in seen: continue
            seen.add(p); rq = hf_get(p)
            if not rq: continue
            last_req = time.time()
            if time.time() - float(rq.get('ts', 0)) > cfg.get('stale', 1800):      # too old: the box already answered with her small brain
                if slot == jslot: hf_put(api, {}, deletes=[p], msg='heavy stale req')
                continue
            pool.submit(judge if slot == jslot else answer, p, rq)
        if time.time() - beat > 90: beat_now('ready'); beat = time.time()
        time.sleep(3)
    beat_now('draining'); pool.shutdown(wait=True); beat_now('ended')
    log('agent ended; requests seen', len(seen), 'reason', 'idle' if time.time() - last_req >= idle else 'max')

if __name__ == '__main__':
    m = sys.argv[1]; a = sys.argv[2] if len(sys.argv) > 2 else ''
    try:
        if m == 'agent': mode_agent(a, sys.argv[3])
        else: {'probe': mode_probe, 'member': mode_member, 'judge': mode_judge, 'session': mode_session}[m](a)
    except Exception:
        import traceback; tb = traceback.format_exc(); print(tb[-1500:], flush=True)
        try: hf_put(hf_api(), {f'heavy/errors/{RUN}.json': {'mode': m, 'tb': tb[-4000:]}}, msg='heavy error')
        except Exception: pass
        raise
    finally:
        for p in PROCS.values():
            try: p.terminate()
            except Exception: pass
