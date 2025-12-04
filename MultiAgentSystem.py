import os
import json
import uuid
import time
import threading
import webbrowser
import logging
import re
from flask import Flask, request, jsonify, Response, stream_with_context

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

class CoreEngine:
    def __init__(self):
        self.model = None
        self.sessions = {}
        self.lock = threading.Lock()
        self.config = {
            "n_gpu_layers": -1,
            "n_ctx": 8192,
            "n_threads": 4,
            "n_batch": 512
        }
        self.has_llama = False
        try:
            from llama_cpp import Llama
            self.has_llama = True
        except: pass

    def update_config(self, cfg):
        self.config.update(cfg)

    def load_model(self, path):
        if not self.has_llama: return False
        with self.lock:
            if self.model: del self.model
            from llama_cpp import Llama
            self.model = Llama(
                model_path=path,
                n_gpu_layers=int(self.config["n_gpu_layers"]),
                n_ctx=int(self.config["n_ctx"]),
                n_threads=int(self.config["n_threads"]),
                n_batch=int(self.config["n_batch"]),
                verbose=False
            )
        return True

    def create_session(self, title="New Operation"):
        uid = str(uuid.uuid4())
        self.sessions[uid] = {
            "id": uid,
            "title": title,
            "history": [],
            "created": time.time()
        }
        return uid

    def get_session(self, uid):
        return self.sessions.get(uid)

    def list_sessions(self):
        return sorted(self.sessions.values(), key=lambda x: x['created'], reverse=True)

    def delete_session(self, uid):
        if uid in self.sessions: del self.sessions[uid]

    def inference(self, prompt, temp=0.1, max_tokens=2048, stop=None):
        if not self.model: return None
        with self.lock:
            full_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
            return self.model.create_completion(
                prompt=full_prompt,
                temperature=temp,
                max_tokens=max_tokens,
                stop=stop or ["###", "User:", "\n\n"],
                echo=False
            )

ENGINE = CoreEngine()

def process_pipeline(original_prompt, session_id, settings):
    sess = ENGINE.get_session(session_id)
    if not sess: return

    def msg(t, c, tgt=None):
        return json.dumps({"type": t, "content": c, "target": tgt}) + "\n"
        
    def clean_resp(text):
        if not text: return ""
        text = text.replace("`", "").replace("'", "").replace('"', "")
        bad = ["SYSTEM:", "GLOBAL CONTEXT:", "Instruction:", "Context:", "Response:"]
        for b in bad:
            text = text.replace(b, "")
        return text.strip()

    def prune_context(ctx, limit=4000):
        if len(ctx) > limit: return "...(truncated)\n" + ctx[-(limit):]
        return ctx

    yield msg("status", "Initializing WHIS-ReAct Protocol...")
    
    global_context = "" 
    if sess['history']:
        global_context += "Prior Chat History:\n"
        for m in sess['history'][-6:]:
            global_context += f"{'User' if m['role']=='user' else 'AI'}: {m['content']}\n"
            
    task_stack = [{"depth": 1, "task": original_prompt, "type": "MAIN"}]
    MAX_DEPTH = 3 
    
    while task_stack:
        item = task_stack.pop() 
        depth = item['depth']
        task = item['task']
        
        prefix = f"[{'>>' * depth}] D{depth}"
        
        if item.get('type') == 'RESUME':
            yield msg("status", f"{prefix} Context Aggregation...")
            yield msg("card", f"Sub-tasks completed for: {task}", f"{prefix} Context")
            continue

        yield msg("status", f"{prefix} WHI Analysis...")

        p_router = f"Input: {task}\nQuestion: Is this a complex task requiring a plan? Answer YES or NO."
        res_router = ENGINE.inference(p_router, temp=0.1, max_tokens=10, stop=["\n"])
        is_complex = "YES" in res_router['choices'][0]['text'].upper() if res_router else True

        if not is_complex:
            val_w = "Chat with user"
            yield msg("card", f"W: {val_w}", f"{prefix} WHI")
            
            val_h = "Reply naturally"
            yield msg("card", f"H: {val_h}", f"{prefix} WHI")
            
            val_i = "YES"
            yield msg("card", f"I: {val_i}", f"{prefix} WHI")
        else:
            p_what = f"Input: {task}\nContext: {prune_context(global_context)}\nTask: Identify the specific goal."
            res_w = ENGINE.inference(p_what, temp=0.1, max_tokens=64, stop=["\n\n", "###"])
            val_w = clean_resp(res_w['choices'][0]['text']) if res_w else ""
            
            if not val_w or "sorry" in val_w.lower() or "understand" in val_w.lower(): 
                val_w = "Execute task"

            yield msg("card", f"W: {val_w}", f"{prefix} WHI")

            p_how = f"Input: {task}\nGoal: {val_w}\nTask: List brief steps to achieve this."
            res_h = ENGINE.inference(p_how, temp=0.1, max_tokens=128, stop=["\n\n", "###"])
            val_h = clean_resp(res_h['choices'][0]['text']) if res_h else ""
            
            if not val_h: val_h = "Execute immediately"

            yield msg("card", f"H: {val_h}", f"{prefix} WHI")

            val_i = "YES"
            if depth < MAX_DEPTH:
                p_is = f"Steps: {val_h}\nQuestion: Is this a single step task? YES or NO."
                res_i = ENGINE.inference(p_is, temp=0.1, max_tokens=10)
                if res_i and "NO" in res_i['choices'][0]['text'].upper(): val_i = "NO"
            
            yield msg("card", f"I: {val_i}", f"{prefix} WHI")

        if val_i == "NO":
            yield msg("status", f"{prefix} Status: COMPLEX. Splitting...")
            
            p_split = f"Goal: {val_w}\nSteps: {val_h}\nTask: Return a JSON list of sub-tasks. Example: [\"Step1\", \"Step2\"]"
            res_split = ENGINE.inference(p_split, temp=0.1)
            
            try:
                txt = res_split['choices'][0]['text']
                match = re.search(r'\[.*\]', txt, re.DOTALL)
                if match:
                    sub_tasks = json.loads(match.group())
                    task_stack.append({"depth": depth, "task": task, "type": "RESUME"})
                    for st in reversed(sub_tasks):
                        task_stack.append({"depth": depth + 1, "task": st, "type": "SUB"})
                    yield msg("card", "\n".join(sub_tasks), f"{prefix} Split Plan")
                    continue
            except:
                val_i = "YES"
                
        if val_i == "YES":
            yield msg("status", f"{prefix} Status: CLEAR. Executing...")
            
            prompt_exec = f"Context: {prune_context(global_context)}\nRequest: {task}\nGoal: {val_w}\nPlan: {val_h}\n\nTask: Write the response now."
            
            with ENGINE.lock:
                stream = ENGINE.model.create_completion(
                    prompt=f"### Instruction:\n{prompt_exec}\n\n### Response:\n",
                    max_tokens=2048,
                    temperature=0.7, 
                    stop=["###"],
                    stream=True
                )
                
                content_chunk = ""
                try:
                    for chunk in stream:
                        txt = chunk['choices'][0]['text']
                        content_chunk += txt
                        yield msg("token", txt)
                except Exception as e:
                    yield msg("token", f"\n[Error: {str(e)}]")
            
            global_context += f"\nUser: {task}\nAI: {content_chunk}\n"
            
    sess['history'].append({"role": "user", "content": original_prompt})
    yield msg("done", "Ready")

HTML_UI = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WHIS-ReAct Studio</title>
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
<style>
:root { --bg: #0f172a; --panel: #1e293b; --border: #334155; --accent: #3b82f6; --text: #f1f5f9; --dim: #94a3b8; }
* { box-sizing: border-box; scrollbar-width: thin; scrollbar-color: #475569 transparent; }
body { margin: 0; font-family: 'JetBrains Mono', 'Segoe UI', monospace; background: var(--bg); color: var(--text); display: flex; height: 100vh; overflow: hidden; font-size: 13px; }

.sidebar { width: 280px; background: var(--panel); border-right: 1px solid var(--border); display: flex; flex-direction: column; }
.brand { padding: 15px; font-weight: 800; color: white; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; }
.brand span { color: var(--accent); }

.main { flex: 1; display: flex; flex-direction: column; }
.chat-area { flex: 1; overflow-y: auto; padding: 30px; display: flex; flex-direction: column; gap: 20px; }
.msg { display: flex; gap: 15px; max-width: 900px; margin: 0 auto; width: 100%; animation: fadeIn 0.3s; }
@keyframes fadeIn { from{opacity:0;translate:0 10px} to{opacity:1;translate:0 0} }
.avatar { width: 32px; height: 32px; background: #334155; border-radius: 4px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; font-weight: bold; }
.ai .avatar { background: var(--accent); color: white; }
.content { flex: 1; line-height: 1.6; background: #1e293b; padding: 15px; border-radius: 8px; border: 1px solid var(--border); }
.content pre { background: #0f172a; padding: 15px; border-radius: 6px; overflow-x: auto; border: 1px solid #334155; }

.input-area { padding: 20px; background: var(--bg); border-top: 1px solid var(--border); }
.input-box { max-width: 900px; margin: 0 auto; background: var(--panel); border: 1px solid var(--border); border-radius: 8px; display: flex; padding: 5px; }
textarea { flex: 1; background: transparent; border: none; color: white; padding: 10px; resize: none; height: 50px; outline: none; font-family: inherit; }
.send-btn { width: 50px; background: var(--accent); border-radius: 6px; display: flex; align-items: center; justify-content: center; cursor: pointer; }

.right-panel { width: 350px; background: var(--panel); border-left: 1px solid var(--border); display: flex; flex-direction: column; }
.panel-head { padding: 15px; font-weight: 700; border-bottom: 1px solid var(--border); color: var(--dim); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
.feed { flex: 1; overflow-y: auto; padding: 15px; display: flex; flex-direction: column; gap: 10px; }
.card { background: #0f172a; border: 1px solid var(--border); border-radius: 4px; overflow: hidden; font-size: 0.8rem; }
.card-head { background: #334155; padding: 6px 10px; font-weight: 700; color: white; display:flex; justify-content:space-between; }
.card-body { padding: 10px; color: #cbd5e1; white-space: pre-wrap; }

.config-area { padding: 15px; overflow-y: auto; flex:1; }
.lbl { font-size: 0.7rem; font-weight: 700; color: var(--dim); margin-top: 15px; margin-bottom: 5px; display: block; }
select, input { width: 100%; background: #0f172a; border: 1px solid #334155; color: white; padding: 8px; border-radius: 4px; outline: none; }
.btn { width: 100%; padding: 10px; background: #334155; color: white; border: none; border-radius: 4px; cursor: pointer; margin-top: 15px; font-weight: 600; }
.btn:hover { background: #475569; }

</style>
</head>
<body>

<div class="sidebar">
    <div class="brand"><span>WHIS-ReAct</span> v2.0</div>
    <div class="config-area">
        <div class="lbl">MODEL (GGUF)</div>
        <select id="modelSelect" onchange="loadSelectedModel()"><option>Scanning...</option></select>
        <div id="modelStatus" style="font-size:0.7rem; color:#666; margin-bottom:5px">Offline</div>

        <div class="lbl">HARDWARE CONFIG</div>
        <input type="number" id="hwGpu" value="-1" placeholder="GPU Layers">
        <div style="height:5px"></div>
        <input type="number" id="hwCtx" value="8192" placeholder="Context Size">
        <button class="btn" onclick="applyHardware()">Apply Config</button>

        <div class="lbl">SESSIONS</div>
        <div id="sessionList"></div>
        <button class="btn" onclick="createNewChat()">New Operation</button>
    </div>
</div>

<div class="main">
    <div class="chat-area" id="chatContainer">
        <div style="text-align:center; margin-top:30vh; opacity:0.3">
            <h1>WHIS-ReAct ENGINE</h1>
            <p>Hard-Coded Logic. Zero Guessing.</p>
        </div>
    </div>
    <div class="input-area">
        <div class="input-box">
            <textarea id="userInput" placeholder="Enter Task for WHIS Dissection..." onkeydown="if(event.key=='Enter'&&!event.shiftKey){event.preventDefault();runPipeline();}"></textarea>
            <div class="send-btn" onclick="runPipeline()"><i class="fas fa-play"></i></div>
        </div>
    </div>
</div>

<div class="right-panel">
    <div class="panel-head">Logic Trace</div>
    <div class="feed" id="protocolFeed"></div>
</div>

<script>
let currentSessionId = null;

window.onload = function() {
    scanForModels();
    refreshSessionList();
}

async function scanForModels() {
    const res = await fetch('/scan');
    const data = await res.json();
    const sel = document.getElementById('modelSelect');
    sel.innerHTML = "<option>Select Model...</option>";
    data.forEach(m => sel.innerHTML += `<option value="${m.path}">${m.name}</option>`);
}

async function loadSelectedModel() {
    const path = document.getElementById('modelSelect').value;
    document.getElementById('modelStatus').innerText = "Loading...";
    await fetch('/load_model', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({path})
    });
    document.getElementById('modelStatus').innerText = "Online";
    document.getElementById('modelStatus').style.color = "#4ade80";
}

async function applyHardware() {
    await fetch('/config_hardware', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            n_gpu_layers: document.getElementById('hwGpu').value,
            n_ctx: document.getElementById('hwCtx').value
        })
    });
    alert("Config Updated. Please reload model.");
}

async function createNewChat() {
    const res = await fetch('/create_session', {method: 'POST'});
    const data = await res.json();
    currentSessionId = data.id;
    refreshSessionList();
    document.getElementById('chatContainer').innerHTML = "";
    document.getElementById('protocolFeed').innerHTML = "";
}

async function refreshSessionList() {
    const res = await fetch('/list_sessions');
    const data = await res.json();
    const list = document.getElementById('sessionList');
    list.innerHTML = "";
    data.forEach(s => {
        const d = document.createElement('div');
        d.style.padding = "5px";
        d.style.cursor = "pointer";
        d.style.color = "#94a3b8";
        d.innerText = "> " + s.title.substring(0, 20);
        d.onclick = () => { currentSessionId = s.id; document.getElementById('chatContainer').innerHTML = ""; };
        list.appendChild(d);
    });
    if(!currentSessionId && data.length) currentSessionId = data[0].id;
}

function appendMessage(role, text) {
    const div = document.createElement('div');
    div.className = `msg ${role}`;
    div.innerHTML = `
        <div class="avatar">${role === 'user' ? 'U' : 'AI'}</div>
        <div class="content">${parseMarkdown(text)}</div>
    `;
    document.getElementById('chatContainer').appendChild(div);
    return div.querySelector('.content');
}

function parseMarkdown(text) {
    if(!text) return "";
    return text
        .replace(/</g, "&lt;").replace(/>/g, "&gt;")
        .replace(/```(\w*)([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
        .replace(/\*\*(.*?)\*\*/g, '<b>$1</b>')
        .replace(/\n/g, '<br>');
}

async function runPipeline() {
    if (!currentSessionId) await createNewChat();
    
    const input = document.getElementById('userInput');
    const text = input.value.trim();
    if (!text) return;
    
    input.value = "";
    appendMessage('user', text);
    const aiBubble = appendMessage('ai', '');
    let fullText = "";

    const res = await fetch('/stream', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({prompt: text, id: currentSessionId, settings: {}})
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, {stream: true});
        const lines = buffer.split("\n"); 
        buffer = lines.pop();

        for (const line of lines) {
            if (!line) continue;
            try {
                const data = JSON.parse(line);
                if (data.type === 'token') {
                    fullText += data.content;
                    aiBubble.innerHTML = parseMarkdown(fullText);
                } else if (data.type === 'card') {
                    const c = document.createElement('div');
                    c.className = 'card';
                    c.innerHTML = `<div class="card-head">${data.target}</div><div class="card-body">${data.content}</div>`;
                    document.getElementById('protocolFeed').prepend(c);
                }
                document.getElementById('chatContainer').scrollTo(0, 99999);
            } catch (e) {}
        }
    }
}
</script>
</body>
</html>
"""

@app.route('/')
def index(): return HTML_UI

@app.route('/scan')
def scan():
    m = []
    try:
        files = os.listdir('.')
        for n in files:
            if n.endswith('.gguf'):
                m.append({'name': n, 'path': os.path.join(os.getcwd(), n)})
    except: pass
    return jsonify(m)

@app.route('/config_hardware', methods=['POST'])
def config_hw():
    ENGINE.update_config(request.json)
    return jsonify({'ok': True})

@app.route('/load_model', methods=['POST'])
def load_model():
    ENGINE.load_model(request.json['path'])
    return jsonify({'ok': True})

@app.route('/list_sessions')
def list_sessions():
    return jsonify([{'id': s['id'], 'title': s['title']} for s in ENGINE.list_sessions()])

@app.route('/create_session', methods=['POST'])
def create_session():
    return jsonify({'id': ENGINE.create_session()})

@app.route('/stream', methods=['POST'])
def stream():
    d = request.json
    return Response(
        stream_with_context(process_pipeline(d['prompt'], d['id'], d['settings'])),
        mimetype='application/x-ndjson'
    )

if __name__ == '__main__':
    threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(port=5000, debug=False, use_reloader=False, threaded=True)
