"""Local web tool for manually cleaning up the ReID dataset.

Dataset folders were found to contain crops from multiple different
physical people mixed together (confirmed via inspect_dataset_mixing.py
contact sheets). This tool lets you browse each dataset/<Name>/ folder's
crops (grouped by their idN_/plain sub-batch, the natural unit for spotting
a mixed-in batch), multi-select thumbnails, and move the selection into a
different (existing or new) person folder or into a "reject" bin -- all as
non-destructive file moves so nothing is ever deleted outright.

Usage:
    python relabel_tool.py --data dataset --port 8765
Then open http://localhost:8765 in a browser.
"""

import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

DATA_ROOT = "dataset"
REJECT_DIR_NAME = "_rejected"

IMG_EXTS = (".jpg", ".jpeg", ".png")


def is_img(fname):
    return fname.lower().endswith(IMG_EXTS)


def list_people():
    people = []
    for d in sorted(os.listdir(DATA_ROOT)):
        p = os.path.join(DATA_ROOT, d)
        if os.path.isdir(p) and d != REJECT_DIR_NAME:
            people.append(d)
    return people


def group_files(person_dir):
    files = sorted(f for f in os.listdir(person_dir) if is_img(f))
    groups = defaultdict(list)
    for f in files:
        m = re.match(r"id(\d+)_", f)
        key = f"id{m.group(1)}" if m else "plain"
        groups[key].append(f)
    return dict(sorted(groups.items(), key=lambda kv: (kv[0] == "plain", kv[0])))


def safe_person_dir(name):
    """Resolve a person folder name to a path, refusing traversal outside DATA_ROOT."""
    name = name.strip()
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError("invalid person name")
    path = os.path.join(DATA_ROOT, name)
    if os.path.commonpath([os.path.abspath(DATA_ROOT), os.path.abspath(path)]) != os.path.abspath(DATA_ROOT):
        raise ValueError("invalid person name")
    return path


def safe_file_in_person(person, fname):
    if fname != os.path.basename(fname) or not is_img(fname):
        raise ValueError("invalid filename")
    person_dir = safe_person_dir(person)
    path = os.path.join(person_dir, fname)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def unique_dest(dest_dir, fname):
    base, ext = os.path.splitext(fname)
    candidate = fname
    n = 1
    while os.path.exists(os.path.join(dest_dir, candidate)):
        candidate = f"{base}_moved{n}{ext}"
        n += 1
    return candidate


INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>ReID Dataset Relabeler</title>
<style>
  body { font-family: system-ui, sans-serif; background: #1e1e1e; color: #ddd; margin: 0; padding: 12px; }
  #topbar { display: flex; gap: 8px; align-items: center; margin-bottom: 12px; flex-wrap: wrap; }
  select, input, button { font-size: 14px; padding: 6px 8px; }
  #groups { display: flex; flex-direction: column; gap: 18px; }
  .group { border: 1px solid #444; border-radius: 6px; padding: 8px; }
  .group h3 { margin: 0 0 8px 0; font-size: 14px; color: #9cf; }
  .grid { display: flex; flex-wrap: wrap; gap: 4px; }
  .thumb-wrap { position: relative; cursor: pointer; border: 3px solid transparent; }
  .thumb-wrap.selected { border-color: #4caf50; }
  .thumb-wrap img { width: 96px; height: 96px; object-fit: cover; display: block; }
  .thumb-wrap .fname { position: absolute; bottom: 0; left: 0; right: 0; font-size: 9px; background: rgba(0,0,0,.6); color: #fff; padding: 1px 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  #actionbar { position: sticky; top: 0; background: #1e1e1e; z-index: 10; padding: 8px 0; border-bottom: 1px solid #444; margin-bottom: 12px; }
  #status { margin-left: 8px; color: #9c9; }
  button { cursor: pointer; }
  button.danger { background: #a33; color: #fff; border: none; border-radius: 4px; }
  button.primary { background: #2a7; color: #fff; border: none; border-radius: 4px; }
</style>
</head>
<body>
<div id="actionbar">
  <div id="topbar">
    <label>Folder: <select id="personSelect"></select></label>
    <button id="refreshBtn">Refresh</button>
    <span>|</span>
    <label>Move selection to: <input id="destInput" list="peopleList" placeholder="existing or new name"></label>
    <datalist id="peopleList"></datalist>
    <button class="primary" id="moveBtn">Move selected</button>
    <button class="danger" id="rejectBtn">Reject selected</button>
    <button id="clearBtn">Clear selection</button>
    <span id="status"></span>
  </div>
</div>
<div id="groups"></div>

<script>
let selected = new Set();
let currentPerson = null;

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    const t = await res.text();
    throw new Error(t || res.statusText);
  }
  return res.json();
}

async function loadPeople() {
  const data = await api('/api/people');
  const sel = document.getElementById('personSelect');
  const dl = document.getElementById('peopleList');
  sel.innerHTML = '';
  dl.innerHTML = '';
  for (const p of data.people) {
    const opt = document.createElement('option');
    opt.value = p; opt.textContent = p;
    sel.appendChild(opt);
    const dopt = document.createElement('option');
    dopt.value = p;
    dl.appendChild(dopt);
  }
  if (!currentPerson && data.people.length) currentPerson = data.people[0];
  sel.value = currentPerson;
}

function thumbEl(person, fname) {
  const wrap = document.createElement('div');
  wrap.className = 'thumb-wrap';
  wrap.dataset.key = person + '/' + fname;
  const img = document.createElement('img');
  img.src = '/crop/' + encodeURIComponent(person) + '/' + encodeURIComponent(fname);
  img.loading = 'lazy';
  const label = document.createElement('div');
  label.className = 'fname';
  label.textContent = fname;
  wrap.appendChild(img);
  wrap.appendChild(label);
  wrap.onclick = () => {
    const key = wrap.dataset.key;
    if (selected.has(key)) { selected.delete(key); wrap.classList.remove('selected'); }
    else { selected.add(key); wrap.classList.add('selected'); }
    updateStatus();
  };
  return wrap;
}

function updateStatus() {
  document.getElementById('status').textContent = selected.size + ' selected';
}

async function loadGroups() {
  currentPerson = document.getElementById('personSelect').value;
  selected.clear();
  updateStatus();
  const data = await api('/api/groups?person=' + encodeURIComponent(currentPerson));
  const container = document.getElementById('groups');
  container.innerHTML = '';
  for (const [key, files] of Object.entries(data.groups)) {
    const g = document.createElement('div');
    g.className = 'group';
    const h = document.createElement('h3');
    h.textContent = key + ' (n=' + files.length + ')';
    g.appendChild(h);
    const grid = document.createElement('div');
    grid.className = 'grid';
    for (const fname of files) grid.appendChild(thumbEl(currentPerson, fname));
    g.appendChild(grid);
    container.appendChild(g);
  }
}

function selectionPayload() {
  const items = [...selected].map(k => {
    const idx = k.indexOf('/');
    return { person: k.slice(0, idx), fname: k.slice(idx + 1) };
  });
  return items;
}

document.getElementById('personSelect').addEventListener('change', loadGroups);
document.getElementById('refreshBtn').addEventListener('click', loadGroups);
document.getElementById('clearBtn').addEventListener('click', () => {
  selected.clear();
  document.querySelectorAll('.thumb-wrap.selected').forEach(e => e.classList.remove('selected'));
  updateStatus();
});

document.getElementById('moveBtn').addEventListener('click', async () => {
  const dest = document.getElementById('destInput').value.trim();
  if (!dest) { alert('Enter a destination folder name'); return; }
  if (selected.size === 0) { alert('Nothing selected'); return; }
  await api('/api/move', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({dest, items: selectionPayload()})
  });
  await loadPeople();
  await loadGroups();
});

document.getElementById('rejectBtn').addEventListener('click', async () => {
  if (selected.size === 0) { alert('Nothing selected'); return; }
  if (!confirm('Move ' + selected.size + ' crop(s) to _rejected/?')) return;
  await api('/api/move', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({dest: '_rejected', items: selectionPayload()})
  });
  await loadGroups();
});

(async () => {
  await loadPeople();
  await loadGroups();
})();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, message, status=400):
        self._send_json({"error": message}, status)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/api/people":
            self._send_json({"people": list_people()})
            return

        if path == "/api/groups":
            qs = parse_qs(parsed.query)
            person = qs.get("person", [None])[0]
            try:
                person_dir = safe_person_dir(person)
            except ValueError as e:
                self._send_error_json(str(e))
                return
            if not os.path.isdir(person_dir):
                self._send_error_json("no such folder", 404)
                return
            self._send_json({"groups": group_files(person_dir)})
            return

        if path.startswith("/crop/"):
            rest = unquote(path[len("/crop/"):])
            parts = rest.split("/", 1)
            if len(parts) != 2:
                self._send_error_json("bad path", 404)
                return
            person, fname = parts
            try:
                full = safe_file_in_person(person, fname)
            except (ValueError, FileNotFoundError):
                self._send_error_json("not found", 404)
                return
            with open(full, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self._send_error_json("not found", 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/move":
            self._send_error_json("not found", 404)
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length))
            dest_name = payload["dest"]
            items = payload["items"]
        except (KeyError, json.JSONDecodeError):
            self._send_error_json("bad request")
            return

        try:
            dest_dir = safe_person_dir(dest_name)
        except ValueError as e:
            self._send_error_json(str(e))
            return
        os.makedirs(dest_dir, exist_ok=True)

        moved, errors = [], []
        for item in items:
            try:
                src = safe_file_in_person(item["person"], item["fname"])
            except (ValueError, FileNotFoundError, KeyError) as e:
                errors.append(f"{item}: {e}")
                continue
            dest_fname = unique_dest(dest_dir, item["fname"])
            shutil.move(src, os.path.join(dest_dir, dest_fname))
            moved.append(item["fname"])

        self._send_json({"moved": moved, "errors": errors})


def main():
    global DATA_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="dataset")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    DATA_ROOT = args.data

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[INFO] Serving {os.path.abspath(DATA_ROOT)} at http://localhost:{args.port}")
    print("[INFO] Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
