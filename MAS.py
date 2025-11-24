import os
import json
import uuid
import time
import threading
import webbrowser
import logging
import sys
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

    def inference(self, prompt, temp=0.7, max_tokens=4096, stop=None, stream=False):
        if not self.model: return None
        sys_p = "<|system|>\nSystem\n<|user|>\n"
        full_prompt = f"{sys_p}{prompt}\n<|assistant|>\n"
        return self.model.create_completion(
            prompt=full_prompt,
            temperature=temp,
            max_tokens=max_tokens,
            stop=stop or ["<|user|>"],
            stream=stream
        )

ENGINE = CoreEngine()

def process_pipeline(prompt, session_id, settings):
    sess = ENGINE.get_session(session_id)
    if not sess: return

    def msg(t, c, tgt=None):
        return json.dumps({"type": t, "content": c, "target": tgt}) + "\n"

    yield msg("status", "Initializing SSESIS v1.0...")

    feats = settings.get('manual_protocols', {})
    
    if settings.get('auto_mode'):
        yield msg("status", "Dynamic Protocol Selection...")
        selector = (
            f"Task: {prompt}\n"
            "Identify required protocols. Output JSON boolean.\n"
            "Keys: facts, deep, topology, plan, debate, simulation, audit.\n"
            "Example: {\"facts\": true, \"deep\": false}\n"
            "JSON:"
        )
        res = ENGINE.inference(selector, temp=0.1)
        if res:
            try:
                txt = res['choices'][0]['text']
                j = json.loads(txt[txt.find('{'):txt.rfind('}')+1])
                feats = j
                yield msg("log", f"Auto-Selected: {', '.join([k for k,v in j.items() if v])}")
            except:
                feats = {'facts': True, 'plan': True, 'simulation': True}

    history_txt = ""
    for m in sess['history'][-4:]:
        r = "User" if m['role'] == 'user' else "Assistant"
        history_txt += f"{r}: {m['content']}\n"

    cfs = ""
    feb = ""

    yield msg("status", "Apollo Integrity Check")

    if feats.get('facts'):
        yield msg("status", "0.0 Parmenides (Facts)")
        r = ENGINE.inference(f"Extract Axioms & Constraints:\n{prompt}", temp=0.1)
        if r:
            cfs = r['choices'][0]['text'].strip()
            yield msg("card", cfs, "Facts")

    if feats.get('deep'):
        yield msg("status", "14.0 Noesis (Deep Analysis)")
        r = ENGINE.inference(f"Deep Analysis:\n{prompt}\nContext: {cfs}", temp=0.6)
        if r: yield msg("card", r['choices'][0]['text'].strip(), "Analysis")

    if feats.get('topology'):
        yield msg("status", "16.0 Utias (Topology)")
        r = ENGINE.inference(f"Dependency Mapping:\n{prompt}", temp=0.2)
        if r: yield msg("card", r['choices'][0]['text'].strip(), "Topology")

    yield msg("status", "0-Phase Prometheus (Blueprint)")
    r = ENGINE.inference(f"Execution Blueprint:\n{prompt}\nFacts: {cfs}", temp=0.3)
    if r:
        feb = r['choices'][0]['text'].strip()
        yield msg("card", feb, "Blueprint")

    final_plan = feb

    if feats.get('debate'):
        yield msg("status", "5.0 Agon (Multi-Agent)")
        yield msg("log", "Strategos Alpha...")
        p1 = ENGINE.inference(f"Safe Plan:\n{feb}", temp=0.2)['choices'][0]['text']
        yield msg("log", "Strategos Beta...")
        p2 = ENGINE.inference(f"Risky Plan:\n{feb}", temp=0.8)['choices'][0]['text']
        yield msg("log", "Dikastes Verdict...")
        v = ENGINE.inference(f"Select Best:\nA: {p1}\nB: {p2}", temp=0.1)
        if v:
            final_plan = v['choices'][0]['text'].strip()
            yield msg("card", final_plan, "Verdict")

    yield msg("status", "19.0 Hephaestus (Execution)")
    
    stream = ENGINE.inference(
        f"Task: {prompt}\nPlan: {final_plan}\nHistory: {history_txt}\nExecute.",
        temp=settings.get('temp', 0.7),
        stream=True
    )
    
    full_resp = ""
    if stream:
        for chunk in stream:
            txt = chunk['choices'][0]['text']
            full_resp += txt
            yield msg("token", txt)

    if feats.get('simulation') and "```" in full_resp:
        yield msg("status", "26.0 Basanos (Simulation)")
        r = ENGINE.inference(f"Simulate code execution. Output Logs:\n{full_resp}", temp=0.1)
        if r:
            logs = r['choices'][0]['text'].strip()
            yield msg("card", logs, "Sim Logs")
            if "Error" in logs or "Traceback" in logs:
                yield msg("status", "4.5 Anakainosis (Repair)")
                f = ENGINE.inference(f"Fix code using logs:\nCode: {full_resp}\nLogs: {logs}", temp=0.1)
                if f:
                    fix = f['choices'][0]['text'].strip()
                    full_resp += f"\n\n### Auto-Repair\n{fix}"
                    yield msg("token", f"\n\n### Auto-Repair\n{fix}")

    if feats.get('audit'):
        yield msg("status", "29.0 Praetorian (Audit)")
        r = ENGINE.inference(f"Security Audit:\n{full_resp}", temp=0.1)
        if r: yield msg("card", r['choices'][0]['text'].strip(), "Audit")

    yield msg("status", "10.0 Kerberos (Final Gateway)")
    sess['history'].append({"role": "user", "content": prompt})
    sess['history'].append({"role": "assistant", "content": full_resp})
    yield msg("done", "Ready")

# NOTE: The 'r' below is what fixes the Javascript syntax error
HTML_UI = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SSESIS Studio v1.0</title>
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
<style>
:root { --bg: #09090b; --panel: #18181b; --border: #27272a; --accent: #8b5cf6; --text: #e4e4e7; --dim: #a1a1aa; }
* { box-sizing: border-box; scrollbar-width: thin; scrollbar-color: #3f3f46 transparent; }
body { margin: 0; font-family: 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); display: flex; height: 100vh; overflow: hidden; font-size: 14px; }

.sidebar { width: 300px; background: var(--panel); border-right: 1px solid var(--border); display: flex; flex-direction: column; flex-shrink: 0; }
.brand { padding: 15px; font-weight: 800; color: white; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; }
.brand span { color: var(--accent); }

.tabs { display: flex; background: #111; border-bottom: 1px solid var(--border); }
.tab { flex: 1; padding: 10px; text-align: center; cursor: pointer; color: var(--dim); font-weight: 600; border-bottom: 2px solid transparent; }
.tab:hover { color: white; background: #222; }
.tab.active { border-color: var(--accent); color: white; background: #222; }

.tab-pane { flex: 1; overflow-y: auto; padding: 15px; display: none; }
.tab-pane.active { display: block; }

.chat-item { padding: 10px; margin-bottom: 5px; border-radius: 6px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; color: var(--dim); }
.chat-item:hover { background: #27272a; color: white; }
.chat-item.active { background: #4c1d95; color: white; border: 1px solid var(--accent); }

.lbl { display: block; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; color: #71717a; margin-bottom: 8px; }
input, select { width: 100%; background: #09090b; border: 1px solid #3f3f46; color: white; padding: 8px; border-radius: 4px; margin-bottom: 10px; outline: none; }
.btn { width: 100%; padding: 10px; background: #27272a; color: white; border: none; border-radius: 6px; cursor: pointer; font-weight: 600; margin-bottom: 10px; }
.btn:hover { background: #3f3f46; }
.btn-primary { background: var(--accent); }
.btn-primary:hover { opacity: 0.9; }

.switch-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
.switch { position: relative; width: 34px; height: 18px; }
.switch input { opacity: 0; width: 0; height: 0; }
.slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: #3f3f46; border-radius: 18px; transition: .3s; }
.slider:before { position: absolute; content: ""; height: 14px; width: 14px; left: 2px; bottom: 2px; background-color: white; border-radius: 50%; transition: .3s; }
input:checked + .slider { background-color: var(--accent); }
input:checked + .slider:before { transform: translateX(16px); }

.main { flex: 1; display: flex; flex-direction: column; position: relative; }
.chat-area { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 20px; }
.msg { display: flex; gap: 15px; max-width: 900px; margin: 0 auto; width: 100%; }
.avatar { width: 32px; height: 32px; background: #27272a; border-radius: 6px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
.ai .avatar { background: var(--accent); color: white; }
.content { flex: 1; line-height: 1.6; overflow-x: hidden; }
.content pre { background: #111; padding: 15px; border-radius: 8px; overflow-x: auto; border: 1px solid #333; }
.content code { background: #27272a; padding: 2px 5px; border-radius: 4px; font-family: monospace; color: #f472b6; }

.input-area { padding: 20px; background: var(--bg); border-top: 1px solid var(--border); }
.input-box { max-width: 900px; margin: 0 auto; background: var(--panel); border: 1px solid #3f3f46; border-radius: 8px; display: flex; padding: 5px; }
textarea { flex: 1; background: transparent; border: none; color: white; padding: 10px; resize: none; height: 40px; max-height: 150px; outline: none; }
.send-btn { width: 40px; height: 40px; background: var(--accent); border-radius: 6px; display: flex; align-items: center; justify-content: center; cursor: pointer; align-self: flex-end; }

.right-panel { width: 320px; background: var(--panel); border-left: 1px solid var(--border); display: flex; flex-direction: column; flex-shrink: 0; }
.panel-head { padding: 15px; font-weight: 700; border-bottom: 1px solid var(--border); color: var(--dim); font-size: 0.8rem; text-transform: uppercase; }
.feed { flex: 1; overflow-y: auto; padding: 15px; display: flex; flex-direction: column; gap: 10px; }
.card { background: #09090b; border: 1px solid #3f3f46; border-radius: 6px; overflow: hidden; animation: slideIn 0.3s; }
@keyframes slideIn { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
.card-head { background: #27272a; padding: 8px 12px; font-weight: 700; color: #a1a1aa; font-size: 0.75rem; border-bottom: 1px solid #3f3f46; }
.card-body { padding: 12px; font-family: monospace; color: #d4d4d8; white-space: pre-wrap; max-height: 250px; overflow-y: auto; font-size: 0.85rem; }

.modal { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); z-index: 1000; display: none; align-items: center; justify-content: center; }
.modal-content { background: var(--panel); width: 600px; max-width: 90%; padding: 25px; border-radius: 10px; border: 1px solid #444; }
.modal-content h2 { margin-top: 0; color: var(--accent); }
.modal-content li { margin-bottom: 8px; color: #ccc; }
</style>
</head>
<body>

<div id="readmeModal" class="modal" onclick="closeReadme()">
    <div class="modal-content" onclick="event.stopPropagation()">
        <h2>SSESIS Studio v1.0</h2>
        <p><b>Production Release</b> | Single-File Cognitive Operating System</p>
        
        <h3>Getting Started</h3>
        <ol>
            <li>Go to <b>Config</b> tab.</li>
            <li>Select a model (GGUF format) and wait for "Online" status.</li>
            <li>Go to <b>Hardw</b> tab to set GPU offload (set to -1 for max speed).</li>
            <li>Create a <b>New Operation</b> in the Chat tab.</li>
        </ol>

        <button class="btn btn-primary" onclick="closeReadme()">Close Documentation</button>
    </div>
</div>

<div class="sidebar">
    <div class="brand">
        <span>SSESIS v1.0</span>
        <i class="fas fa-book" style="cursor:pointer; opacity:0.8" onclick="openReadme()"></i>
    </div>
    <div class="tabs">
        <div class="tab active" onclick="switchTab(0)">Chat</div>
        <div class="tab" onclick="switchTab(1)">Config</div>
        <div class="tab" onclick="switchTab(2)">Hardw</div>
    </div>

    <div id="tab0" class="tab-pane active">
        <button class="btn" onclick="createNewChat()">+ New Operation</button>
        <div id="sessionList" style="margin-top:10px"></div>
    </div>

    <div id="tab1" class="tab-pane">
        <div class="lbl">Model File</div>
        <select id="modelSelect" onchange="loadSelectedModel()"><option>Scanning...</option></select>
        <div id="modelStatus" style="font-size:0.7rem; color:#666; margin-bottom:15px">Offline</div>
        
        <div class="switch-row">
            <span class="lbl">Dynamic Selection</span>
            <label class="switch"><input type="checkbox" id="autoMode" onchange="toggleManualControls()" checked><span class="slider"></span></label>
        </div>
        
        <div id="manualControls" style="opacity:0.5; pointer-events:none;">
            <div class="switch-row"><span class="lbl">Facts (Parmenides)</span> <label class="switch"><input type="checkbox" id="p_facts"><span class="slider"></span></label></div>
            <div class="switch-row"><span class="lbl">Deep Think (Noesis)</span> <label class="switch"><input type="checkbox" id="p_deep"><span class="slider"></span></label></div>
            <div class="switch-row"><span class="lbl">Topology (Utias)</span> <label class="switch"><input type="checkbox" id="p_topo"><span class="slider"></span></label></div>
            <div class="switch-row"><span class="lbl">Debate (Agon)</span> <label class="switch"><input type="checkbox" id="p_debate"><span class="slider"></span></label></div>
            <div class="switch-row"><span class="lbl">Code Sim (Basanos)</span> <label class="switch"><input type="checkbox" id="p_sim"><span class="slider"></span></label></div>
            <div class="switch-row"><span class="lbl">Audit (Praetorian)</span> <label class="switch"><input type="checkbox" id="p_audit"><span class="slider"></span></label></div>
        </div>
        
        <div class="lbl">Temperature</div>
        <input type="number" id="tempParam" value="0.7" step="0.1">
    </div>

    <div id="tab2" class="tab-pane">
        <div class="lbl">GPU Layers (-1 = All)</div>
        <input type="number" id="hwGpu" value="-1">
        <div class="lbl">Context Size</div>
        <input type="number" id="hwCtx" value="8192">
        <div class="lbl">Threads</div>
        <input type="number" id="hwThreads" value="4">
        <button class="btn btn-primary" onclick="applyHardware()">Apply Settings</button>
    </div>
</div>

<div class="main">
    <div class="chat-area" id="chatContainer">
        <div style="text-align:center; margin-top:20vh; color:#333;">
            <i class="fas fa-microchip" style="font-size:4rem; margin-bottom:20px;"></i><br>
            SYSTEM READY
        </div>
    </div>
    <div class="input-area">
        <div class="input-box">
            <textarea id="userInput" placeholder="Enter Directive..." oninput="this.style.height='auto';this.style.height=this.scrollHeight+'px'" onkeydown="if(event.key=='Enter'&&!event.shiftKey){event.preventDefault();runPipeline();}"></textarea>
            <div class="send-btn" onclick="runPipeline()"><i class="fas fa-arrow-up"></i></div>
        </div>
    </div>
</div>

<div class="right-panel">
    <div class="panel-head">Protocol Activity</div>
    <div class="feed" id="protocolFeed"></div>
</div>

<script>
let currentSessionId = null;

// Ensure this runs immediately
switchTab(0);

window.onload = function() {
    scanForModels();
    refreshSessionList();
}

function switchTab(n) {
    document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
    
    const panes = document.querySelectorAll('.tab-pane');
    const tabs = document.querySelectorAll('.tab');
    
    if(panes[n]) panes[n].classList.add('active');
    if(tabs[n]) tabs[n].classList.add('active');
}

function toggleManualControls() {
    const isAuto = document.getElementById('autoMode').checked;
    const panel = document.getElementById('manualControls');
    panel.style.opacity = isAuto ? "0.5" : "1";
    panel.style.pointerEvents = isAuto ? "none" : "auto";
}

function openReadme() { document.getElementById('readmeModal').style.display = 'flex'; }
function closeReadme() { document.getElementById('readmeModal').style.display = 'none'; }

async function scanForModels() {
    try {
        const res = await fetch('/scan');
        const data = await res.json();
        const sel = document.getElementById('modelSelect');
        sel.innerHTML = "<option>Select Model...</option>";
        data.forEach(m => sel.innerHTML += `<option value="${m.path}">${m.name}</option>`);
    } catch(e) { console.error(e); }
}

async function loadSelectedModel() {
    const path = document.getElementById('modelSelect').value;
    if (!path || path.includes("Select")) return;
    
    const status = document.getElementById('modelStatus');
    status.innerText = "Loading...";
    
    await fetch('/load_model', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({path})
    });
    
    status.innerText = "Online";
    status.style.color = "#00ff88";
}

async function applyHardware() {
    const cfg = {
        n_gpu_layers: document.getElementById('hwGpu').value,
        n_ctx: document.getElementById('hwCtx').value,
        n_threads: document.getElementById('hwThreads').value
    };
    await fetch('/config_hardware', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(cfg)
    });
    loadSelectedModel(); 
}

async function refreshSessionList() {
    try {
        const res = await fetch('/list_sessions');
        const data = await res.json();
        const list = document.getElementById('sessionList');
        list.innerHTML = "";
        
        data.forEach(s => {
            const el = document.createElement('div');
            el.className = `chat-item ${currentSessionId === s.id ? 'active' : ''}`;
            el.innerHTML = `<span>${s.title.substring(0, 20)}</span> <i class="fas fa-trash" onclick="deleteChat('${s.id}', event)"></i>`;
            el.onclick = () => loadChat(s.id);
            list.appendChild(el);
        });
        
        if (!currentSessionId && data.length > 0) loadChat(data[0].id);
    } catch(e) { console.error(e); }
}

async function createNewChat() {
    const res = await fetch('/create_session', {method: 'POST'});
    const data = await res.json();
    loadChat(data.id);
}

async function deleteChat(id, e) {
    e.stopPropagation();
    if(!confirm("Delete this operation?")) return;
    await fetch('/delete_session', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({id})
    });
    if (currentSessionId === id) currentSessionId = null;
    refreshSessionList();
}

async function loadChat(id) {
    currentSessionId = id;
    
    document.querySelectorAll('.chat-item').forEach(el => el.classList.remove('active'));
    refreshSessionList();
    
    const res = await fetch('/get_history', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({id})
    });
    const data = await res.json();
    
    document.getElementById('chatContainer').innerHTML = "";
    document.getElementById('protocolFeed').innerHTML = "";
    
    data.history.forEach(msg => appendMessage(msg.role, msg.content));
}

function appendMessage(role, text) {
    const div = document.createElement('div');
    div.className = `msg ${role === 'user' ? 'user' : 'ai'}`;
    div.innerHTML = `
        <div class="avatar"><i class="fas fa-${role === 'user' ? 'user' : 'robot'}"></i></div>
        <div class="content">${parseMarkdown(text)}</div>
    `;
    const container = document.getElementById('chatContainer');
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    return div.querySelector('.content');
}

function parseMarkdown(text) {
    if(!text) return "";
    return text
        .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
        .replace(/```(\w*)([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>')
        .replace(/\n/g, '<br>');
}

async function runPipeline() {
    if (!currentSessionId) await createNewChat();
    
    const input = document.getElementById('userInput');
    const text = input.value.trim();
    if (!text) return;
    
    input.value = "";
    input.style.height = '40px';
    
    appendMessage('user', text);
    const aiBubble = appendMessage('ai', 'Thinking...');
    let fullText = "";

    const settings = {
        auto_mode: document.getElementById('autoMode').checked,
        temp: parseFloat(document.getElementById('tempParam').value),
        manual_protocols: {
            facts: document.getElementById('p_facts').checked,
            deep: document.getElementById('p_deep').checked,
            topology: document.getElementById('p_topo').checked,
            debate: document.getElementById('p_debate').checked,
            simulation: document.getElementById('p_sim').checked,
            audit: document.getElementById('p_audit').checked
        }
    };

    const res = await fetch('/stream', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({prompt: text, id: currentSessionId, settings})
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
                    const card = document.createElement('div');
                    card.className = 'card';
                    card.innerHTML = `<div class="card-head">${data.target}</div><div class="card-body">${data.content}</div>`;
                    document.getElementById('protocolFeed').prepend(card);
                }
                document.getElementById('chatContainer').scrollTop = document.getElementById('chatContainer').scrollHeight;
            } catch (e) {
                console.log("JSON Parse Error", e);
            }
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
    except Exception as e: 
        print(e)
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

@app.route('/delete_session', methods=['POST'])
def delete_session():
    ENGINE.delete_session(request.json['id'])
    return jsonify({'ok': True})

@app.route('/get_history', methods=['POST'])
def get_history():
    s = ENGINE.get_session(request.json['id'])
    return jsonify({'history': s['history'] if s else []})

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