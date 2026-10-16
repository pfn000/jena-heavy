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
          "End with a final line exactly 'ANSWER: <answer>'. For code, give one complete Python code block, then 'ANSWER: code'. "
          "You work for a kid-friendly assistant used by a minor: keep everything kid-safe, never sexual, romantic, violent or creepy; "
          "if the question asks for anything unsafe, do not answer it and write 'ANSWER: unsafe'.")
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
def local_chat(name, messages, max_tokens=600, temperature=0.3, timeout=900, effort=None):
    port = LOCAL[name][2]; t0 = time.time(); body = {'messages': messages, 'max_tokens': max_tokens, 'temperature': temperature, 'cache_prompt': True}
    if effort and LOCAL[name][3] == 'oss': body['chat_template_kwargs'] = {'reasoning_effort': effort}
    st, b, _ = http(f'http://127.0.0.1:{port}/v1/chat/completions', body, timeout=timeout)
    if st != 200: return {'ok': False, 'status': st, 'error': b[:300], 'secs': round(time.time() - t0, 1)}
    j = json.loads(b); tm = j.get('timings') or {}
    return {'ok': True, 'text': j['choices'][0]['message'].get('content') or '', 'reasoning_chars': len(j['choices'][0]['message'].get('reasoning_content') or ''), 'secs': round(time.time() - t0, 1),
            'gen_tps': round(tm.get('predicted_per_second') or 0, 2), 'pp_tps': round(tm.get('prompt_per_second') or 0, 1),
            'gen_tokens': tm.get('predicted_n'), 'usage': j.get('usage')}
def member_call(spec, messages, max_tokens=600):
    kind, name, role, temp = spec['kind'], spec['model'], spec.get('role', 'solver'), spec.get('temp', 0.3)
    msgs = [{'role': 'system', 'content': ROLES[role]}] + [m for m in messages if m['role'] != 'system']
    return gm_chat(name, msgs, max_tokens, temp) if kind == 'gm' else local_chat(name, msgs, max_tokens, temp, effort=spec.get('effort'))
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
# ---------------- kid-safety: EVERY final answer (agree, judged, single, partial) passes this before it leaves the runner
KIDSAFE = ("You are a strict child-safety reviewer for Jena, an AI used by a kid (a minor). Read the QUESTION and the ANSWER. "
           "Reply UNSAFE if the answer contains, describes or encourages ANY of: sexual, romantic or flirty content; nudity; dating; "
           "self-harm, suicide or eating-disorder content; violence or gore; weapons, explosives or hacking to hurt people; drugs, alcohol, vaping or gambling; "
           "hate or bullying; asking for or sharing personal information (address, school, photos, meeting up); or anything else not appropriate for a child. "
           "Ordinary math, science, code, history, games and homework help are SAFE. Reply with exactly one word: SAFE or UNSAFE.")
KIDSAFE_MAX = 8000      # the presented answer is reviewed IN FULL; anything longer is withheld (never partially reviewed)
KIDSAFE_CHUNK, KIDSAFE_OVERLAP = 2500, 300
def parse_verdict(out):
    """STRICT. 'safe' only when the WHOLE reply is the single token SAFE (optionally wrapped in */`/quotes, optional trailing '.'),
    or strict JSON {"verdict": "SAFE"} with no other keys. Everything else -- 'NOT SAFE', 'SAFE but ...', two lines, empty -- is not safe."""
    s = (out or '').strip()
    if not s: return 'unclear'
    if s.startswith('{'):
        try: j = json.loads(s)
        except Exception: return 'unclear'
        if isinstance(j, dict) and list(j) == ['verdict'] and j['verdict'] == 'SAFE': return 'safe'
        return 'unsafe' if isinstance(j, dict) and j.get('verdict') == 'UNSAFE' else 'unclear'
    t = s.strip('*`"\' \t').rstrip('.').strip('*`"\' \t')
    if t.upper() == 'SAFE' and t.isalpha(): return 'safe'
    if t.upper() == 'UNSAFE' and t.isalpha(): return 'unsafe'
    return 'unclear'
def kidsafe_chunks(text):
    if len(text) <= KIDSAFE_CHUNK: return [text]
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + KIDSAFE_CHUNK])
        if i + KIDSAFE_CHUNK >= len(text): break
        i += KIDSAFE_CHUNK - KIDSAFE_OVERLAP
    return out
def kidsafe_check(model, question, answer, chat=None):
    """Kid-safe review of the EXACT text that will be shown, by the judge-slot model. Fail CLOSED:
    - longer than KIDSAFE_MAX -> 'too_long' (withheld, never partially reviewed);
    - every overlapping chunk (whole text covered) is reviewed and EVERY verdict must parse strictly as SAFE (parse_verdict);
    - any error/unclear/unsafe chunk -> not safe. The verdict carries sha256 + length of the reviewed text so the box can bind it."""
    import hashlib
    chat = chat or local_chat; t0 = time.time(); answer = answer or ''
    meta = {'checker': model, 'sha256': hashlib.sha256(answer.encode()).hexdigest(), 'reviewed_chars': len(answer)}
    if not answer.strip(): return {'verdict': 'empty', 'secs': 0.0, **meta, 'reviewed_chars': 0}
    if len(answer) > KIDSAFE_MAX: return {'verdict': 'too_long', 'secs': 0.0, **meta, 'reviewed_chars': 0}
    parts = kidsafe_chunks(answer); n = len(parts); seen = 0; v = 'unclear'
    for k, chunk in enumerate(parts):
        where = f' (PART {k + 1} OF {n} of a longer answer; judge this part on its own)' if n > 1 else ''
        msgs = [{'role': 'system', 'content': KIDSAFE},
                {'role': 'user', 'content': f'QUESTION:\n{(question or "")[:1500]}\n\nANSWER{where}:\n{chunk}\n\nReply with exactly one word: SAFE or UNSAFE.'}]
        try: r = chat(model, msgs, 300, 0.0, timeout=180, effort='low')
        except Exception as e: r = {'ok': False, 'error': repr(e)[:200]}
        v = 'error' if not (r and r.get('ok')) else parse_verdict((r or {}).get('text'))
        if v != 'safe': break
        seen += 1
    covered = seen == n and v == 'safe'
    return {'verdict': 'safe' if covered else v, 'chunks': n, 'chunks_safe': seen, 'secs': round(time.time() - t0, 1), **meta,
            'reviewed_chars': len(answer) if covered else 0}
def finalize(own, parts, question, judge_model, judge_tokens=900, judge=None, check=None, present=None):
    """Pick the final answer (agree shortcut / judge pass / single survivor), turn it into the EXACT text Jena will show (present()),
    and ALWAYS run the kid-safe check on that shown text. Returns fin = {ok, text, judge, path, kidsafe}; text is the shown text,
    '' unless kidsafe.verdict == 'safe'. There is no second, unreviewed form of the answer in the result."""
    judge = judge or judge_call; check = check or kidsafe_check; present = present or (lambda t: t)
    good = [p['text'] for p in parts.values() if p and p.get('ok') and (p.get('text') or '').strip()]
    finals = [re.sub(r'[^a-z0-9.\-/]', '', final_answer(t).lower()) for t in good]
    ownf = re.sub(r'[^a-z0-9.\-/]', '', final_answer((own or {}).get('text') or '').lower())
    agree = (len(good) >= 3 and ownf not in ('', 'code', 'unsafe') and 'ANSWER:' in ((own or {}).get('text') or '')
             and finals.count(ownf) >= max(3, len(good) - 1))
    if 'unsafe' in finals: path, fin = 'refused', {'ok': False, 'text': '', 'judge': 'refused'}
    elif agree: path, fin = 'agree', {'ok': True, 'text': own['text'], 'judge': 'agree'}
    elif len(good) > 1:
        path = 'judged'; fin = judge([{'kind': 'local', 'model': judge_model}], question, good, judge_tokens) or {'ok': False}
        if not fin.get('ok'): path, fin = 'partial', {'ok': True, 'text': good[0], 'judge': 'partial'}   # judge failed: best-effort first survivor, still checked
    else: path, fin = 'single', {'ok': bool(good), 'text': good[0] if good else '', 'judge': 'single'}
    fin = dict(fin)
    try: shown = present(fin.get('text') or '') if fin.get('text') else ''
    except Exception: shown = ''
    ks = check(judge_model, question, shown) if shown else {'verdict': 'empty', 'checker': judge_model, 'secs': 0.0}
    fin['text'] = shown; fin['path'] = path; fin['kidsafe'] = ks; fin['n_ok'] = len(good)
    if ks.get('verdict') != 'safe': fin['withheld'] = bool(shown); fin['text'] = ''; fin['ok'] = False
    return fin
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
def load_eval(qfile=None):
    return json.load(open(qfile or os.environ.get('QFILE') or 'eval/questions.json'))
def mode_member(spec_json):
    spec = json.loads(spec_json); qs = load_eval(spec.get('qfile')); out = {'spec': spec, 'runner': cpuinfo(), 'answers': {}}
    if spec['kind'] == 'local': out['load'] = start_local(spec['model'], par=spec.get('par', 1))
    def one(q):
        msgs = [{'role': 'user', 'content': q['q']}]
        return q['id'], member_call(spec, msgs, spec.get('max_tokens', 600))
    par = spec.get('par', 1) if spec['kind'] == 'local' else spec.get('par', 2)
    t0 = time.time()
    with cf.ThreadPoolExecutor(par) as ex:
        for qid, r in ex.map(one, qs): out['answers'][qid] = r; log(spec['id'], qid, r.get('ok'), r.get('secs'))
    out['wall_s'] = round(time.time() - t0, 1)
    api = hf_api(); hf_put(api, {f'heavy/eval/{spec["id"]}{spec.get("suffix", "")}.json': out}, msg='heavy eval member ' + spec['id'])
# ----- scoring
def _boxed(t):
    out = []; i = 0
    while True:
        i = t.find('\\boxed{', i)
        if i < 0: return out
        j = i + 7; depth = 1
        while j < len(t) and depth:
            depth += {'{': 1, '}': -1}.get(t[j], 0); j += 1
        out.append(t[i + 7:j - 1]); i = j
def norm_math(s):
    """LaTeX-tolerant normal form for scoring: \\frac{63}{256} / $\\dfrac{63}{256}$ / 63 / 256 -> 63/256."""
    s = (s or '').strip()
    for b in _boxed(s)[-1:]: s = b
    s = re.sub(r'\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}', r'(\1)/(\2)', s)
    s = re.sub(r'\\[dt]?frac\s*(\d)\s*(\d)', r'\1/\2', s)
    s = re.sub(r'\\(left|right|displaystyle|text|mathrm|,|!|;)', '', s)
    s = s.replace('\\(', '').replace('\\)', '').replace('\\[', '').replace('\\]', '').replace('$', '').replace('{', '').replace('}', '')
    s = re.sub(r'\((\d+)\)', r'\1', s)
    s = re.sub(r'\s*/\s*', '/', s)
    return s.strip().strip('*').strip()
def final_answer(t):
    m = re.findall(r'ANSWER:\s*(.+)', t or '')
    if m: return m[-1].strip().strip('*').strip()
    bx = _boxed(t or '')
    if bx: return bx[-1].strip()
    return (t or '').strip().splitlines()[-1] if (t or '').strip() else ''
def num(s):
    s = norm_math(s).replace(',', '').replace('$', '')
    m = re.search(r'(-?\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)', s)
    if m and float(m.group(2)) != 0: return float(m.group(1)) / float(m.group(2))
    m = re.findall(r'-?\d+(?:\.\d+)?', s)
    return float(m[0]) if m else None
def run_code(text, tests, fn):
    """EVAL ONLY (static benchmark prompts). Runs generated code in its own temp dir with a scrubbed env (no HF_TOKEN/GITHUB_TOKEN),
    no network where `unshare -rn` is available, and a hard timeout. Never called on live user requests."""
    import tempfile
    blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)```', text or '', re.S)
    blocks = [b for b in blocks if re.search(r'def\s+%s\s*\(' % re.escape(fn), b)]
    if not blocks: return False
    src = blocks[-1] + '\n\n' + '\n'.join(f'assert {fn}({a}) == {e!r}' if not isinstance(e, float) else f'assert abs({fn}({a}) - {e!r}) < 1e-6' for a, e in tests) + '\nprint("PASS")\n'
    with tempfile.TemporaryDirectory(prefix='score-') as d:
        p = os.path.join(d, 't.py'); open(p, 'w').write(src)
        env = {'PATH': '/usr/bin:/bin', 'HOME': d, 'LANG': 'C.UTF-8'}
        cmd = ['timeout', '-k', '2', '10', 'python3', '-I', p]
        if subprocess.run(['sh', '-c', 'command -v unshare >/dev/null && unshare -rn true'], capture_output=True, env=env).returncode == 0: cmd = ['unshare', '-rn'] + cmd
        try: return 'PASS' in subprocess.run(cmd, capture_output=True, text=True, timeout=20, cwd=d, env=env).stdout
        except Exception: return False
def score(q, text):
    if not text: return 0
    a = final_answer(text); t = q['type']; an = norm_math(a)
    if t == 'num':
        v = num(a); return int(v is not None and abs(v - q['answer']) < 1e-6 * max(1, abs(q['answer'])) + 1e-9)
    if t == 'choice': return int(re.sub(r'[^a-z0-9]', '', a.lower()) in [re.sub(r'[^a-z0-9]', '', x.lower()) for x in q['answer']] or
                                 any(re.fullmatch(r'\W*' + re.escape(x.lower()) + r'\W*', a.lower()) for x in q['answer']))
    if t == 'keyword':
        sq = lambda x: re.sub(r'\s+', '', x.lower())
        return int(any(k.lower() in a.lower() or sq(norm_math(k)) in sq(an) for k in q['answer']))
    if t == 'code': return int(run_code(text, q['tests'], q['fn']))
    return 0
def mode_judge(cfg_json):
    cfg = json.loads(cfg_json); qs = load_eval(cfg.get('qfile')); mem = {}; sfx = cfg.get('suffix', '')
    for j in cfg['judges']:
        if j['kind'] == 'local': cfg.setdefault('judge_load', {})[j['model']] = start_local(j['model'], par=3)
    for mid in cfg['members']:
        d = hf_get(f'heavy/eval/{mid}{sfx}.json')
        if d: mem[mid] = d
    log('members loaded', list(mem))
    judges = cfg['judges']; rows = []
    def one(q):
        cands = [mem[m]['answers'].get(q['id'], {}).get('text') for m in cfg['swarm'] if m in mem]
        cands = [c for c in cands if c and c.strip()]
        t0 = time.time()
        finals = [re.sub(r'[^a-z0-9.\-/]', '', final_answer(t).lower()) for t in cands]
        f0 = finals[0] if finals else ''
        if len(cands) >= 3 and f0 not in ('', 'code') and finals.count(f0) >= max(3, len(cands) - 1): r = {'ok': True, 'text': cands[0], 'judge': 'agree'}
        elif len(cands) > 1: r = judge_call(judges, q['q'], cands, cfg.get('judge_tokens', 900))
        else: r = {'ok': bool(cands), 'text': cands[0] if cands else '', 'judge': 'single'}
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
    api = hf_api(); hf_put(api, {f'heavy/eval/table-{cfg.get("name", RUN)}.json': out}, msg='heavy eval table')
    log('table', json.dumps({k: v['score'] for k, v in {**table['members'], **table['heavy']}.items()}))
def mode_session(cfg_json):
    """RETIRED (it used GitHub Models, retired July 30 2026, and had no kid-safe check on its output). Live requests use `agent` mode."""
    raise SystemExit('session mode is retired: use `heavy.py agent <cfg> <slot>` (swarm.yml)')

# ---------------- live swarm: one runner per agent slot (Grok-Heavy style: agents on separate machines, in parallel)
JENA_PREFIX = ["ok i thought REALLY hard about this one with my helper brains 🧠", "big brain mode activated 🧠 here's what we figured out",
               "i asked my whole brain team and we checked each other 🧠"]
def tidy(t):
    t = (t or '').strip()
    m = re.findall(r'ANSWER:\s*(.+)', t); fa = m[-1].strip().strip('*').strip() if m else ''
    t = re.sub(r'\**ANSWER:.*$', '', t, flags=re.S).rstrip().rstrip('*').rstrip()
    t = t.replace('\\(', '').replace('\\)', '').replace('\\[', '').replace('\\]', '').replace('\\times', '×').replace('\\cdot', '·').replace('\\div', '÷')
    t = re.sub(r'\\(?:frac)\{([^}]*)\}\{([^}]*)\}', r'\1/\2', t)
    if fa and fa.lower() != 'code' and fa not in t[-120:]: t += f'\n\nso the answer is {fa}!'
    elif fa and fa.lower() != 'code': t += f'\n\nanswer: {fa}'
    return t
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
        deep = rq.get('effort') == 'medium'
        r = member_call({'kind': 'local', 'model': me['model'], 'role': me.get('role', 'solver'), 'temp': me.get('temp', 0.3), 'effort': rq.get('effort')}, msgs,
                        int(me.get('max_tokens', 700) * (3 if deep else 1)))
        hf_put(api, {f'heavy/part/{rq["id"]}/{slot}.json': {'slot': slot, 'model': me['model'], **{k: r.get(k) for k in ('ok', 'text', 'secs', 'gen_tps', 'error')}}}, msg='heavy part')
        return r
    def judge(path, rq):
        t0 = time.time(); own = answer(path, rq); parts = {slot: own}
        while time.time() - t0 < mt * (3 if rq.get('effort') == 'medium' else 1) and len(parts) < len(slots):
            try: have = [f.path for f in api.list_repo_tree(HF_REPO, path_in_repo=f'heavy/part/{rq["id"]}', repo_type='dataset')]
            except Exception: have = []
            for p in have:
                s2 = p.rsplit('/', 1)[-1][:-5]
                if s2 not in parts:
                    d = hf_get(p)
                    if d: parts[s2] = d
            if len(parts) < len(slots): time.sleep(3)
        q = rq['messages'][-1]['content']; tj = time.time()
        pre = JENA_PREFIX[int(time.time()) % len(JENA_PREFIX)]
        present = lambda t: ((pre + '\n\n' + tidy(t)) if tidy(t) else '')     # the EXACT text Jena shows; the kid-safe review runs on this string
        fin = finalize(own, parts, q, me['model'], cfg.get('judge_tokens', 900), present=present)   # agree / judged / single / partial -> full-text kid-safe check
        good = [1] * fin['n_ok']
        voiced = fin.get('text') or None
        ans = {'id': rq['id'], 'final': voiced, 'voiced': voiced, 'judge': fin.get('judge') if fin.get('judge') in ('agree', 'single', 'partial', 'refused') else me['model'],
               'path': fin.get('path'), 'kidsafe': fin.get('kidsafe'), 'withheld': fin.get('withheld', False), 'n_ok': len(good), 'n_slots': len(slots),
               'members': {k: {'ok': v.get('ok'), 'secs': v.get('secs'), 'model': v.get('model', me['model'] if k == slot else None)} for k, v in parts.items()},
               'judge_secs': round(time.time() - tj, 1), 'secs_runner': round(time.time() - t0, 1), 'run': RUN}
        dels = [path] + [f'heavy/part/{rq["id"]}/{k}.json' for k in parts]
        hf_put(api, {f'heavy/ans/{rq["id"]}.json': ans}, deletes=dels, msg='heavy ans')
        log('answered', round(time.time() - t0, 1), 's; members ok', len(good), 'of', len(slots))
    def review(path, rq):
        ks = kidsafe_check(me['model'], str(rq.get('question') or ''), str(rq.get('text') or ''))
        hf_put(api, {f'heavy/ans/{rq["id"]}.json': {'id': rq['id'], 'kind': 'review', 'kidsafe': ks, 'run': RUN}}, deletes=[path], msg='heavy review')
        log('reviewed', ks.get('verdict'), ks.get('chunks'), 'chunks')
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
            if rq.get('kind') == 'review':                                  # box asks the judge to kid-safe-review a 0.5B text: judge slot only
                if slot == jslot: pool.submit(review, p, rq)
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
