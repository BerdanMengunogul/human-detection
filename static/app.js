const POLL_INTERVAL_MS = 1000;

const tabBtns = Array.from(document.querySelectorAll(".tab-btn"));

function activateTab(btn) {
  tabBtns.forEach((b) => {
    b.classList.remove("active");
    b.setAttribute("aria-selected", "false");
  });
  document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
  btn.classList.add("active");
  btn.setAttribute("aria-selected", "true");
  document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
  onTabActivated(btn.dataset.tab);
}

tabBtns.forEach((btn, i) => {
  btn.addEventListener("click", () => activateTab(btn));
  btn.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    e.preventDefault();
    const next = tabBtns[(i + (e.key === "ArrowRight" ? 1 : -1) + tabBtns.length) % tabBtns.length];
    next.focus();
    activateTab(next);
  });
});

function setStatValue(id, value) {
  const el = document.getElementById(id);
  if (el.textContent !== String(value)) {
    el.textContent = value;
    el.classList.remove("pulse");
    void el.offsetWidth;
    el.classList.add("pulse");
  }
}

function setConnected(ok) {
  const indicator = document.getElementById("conn-indicator");
  indicator.classList.toggle("offline", !ok);
  indicator.lastChild.textContent = ok ? "Live" : "Offline";
}

let toastTimer = null;
function showToast(message) {
  const toast = document.getElementById("toast");
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 2500);
}

function renderOccupancy(data) {
  setConnected(true);

  setStatValue("occ-count", data.count);
  setStatValue("occ-fps", data.fps);
  setStatValue("occ-status", data.detector_running ? "Running" : "Stopped");
  setStatValue("occ-unique", data.unique_count);
  setStartStopUI(data.detector_running);

  const tbody = document.querySelector("#occ-table tbody");
  tbody.innerHTML = "";
  data.people.forEach((p, i) => {
    const tr = document.createElement("tr");
    tr.style.animationDelay = `${i * 30}ms`;
    const statusClass = p.status === "IN" ? "status-in" : "status-out";
    const label = p.name || `Person-${p.person_id}`;
    const tdName = document.createElement("td");
    tdName.textContent = label;
    const tdStatus = document.createElement("td");
    tdStatus.className = statusClass;
    tdStatus.textContent = p.status;
    const tdSince = document.createElement("td");
    tdSince.textContent = p.since;
    tr.append(tdName, tdStatus, tdSince);
    tbody.appendChild(tr);
  });
  document.getElementById("occ-table").hidden = data.people.length === 0;
  document.getElementById("occ-empty").hidden = data.people.length !== 0;
}

async function refreshOccupancy() {
  try {
    const res = await fetch("/api/occupancy");
    renderOccupancy(await res.json());
  } catch (err) {
    console.error("occupancy poll failed", err);
    setConnected(false);
  }
}

let knownEventIds = new Set();

async function refreshHistory() {
  try {
    const res = await fetch("/api/events?limit=50");
    const data = await res.json();

    const tbody = document.querySelector("#hist-table tbody");

    // Events come back newest-first. If the DB was reset (or this is the
    // first load), the previous rows no longer apply, so start clean.
    const incomingIds = new Set(data.events.map((e) => e.id));
    const isReset = knownEventIds.size > 0 && data.events.length > 0 && ![...incomingIds].some((id) => knownEventIds.has(id));
    if (isReset || data.events.length === 0) {
      tbody.innerHTML = "";
      knownEventIds = new Set();
    }

    let newCount = 0;
    data.events.forEach((e) => {
      if (knownEventIds.has(e.id)) return;
      newCount += 1;
      knownEventIds.add(e.id);
      const tr = document.createElement("tr");
      tr.dataset.eventId = e.id;
      tr.style.animationDelay = `${newCount * 20}ms`;
      const label = e.name || `Person-${e.person_id}`;
      const tdTime = document.createElement("td");
      tdTime.textContent = e.timestamp;
      const tdName = document.createElement("td");
      tdName.textContent = label;
      const tdType = document.createElement("td");
      tdType.textContent = e.event_type.toUpperCase();
      tr.append(tdTime, tdName, tdType);
      tbody.insertBefore(tr, tbody.firstChild);
    });

    document.getElementById("hist-table").hidden = data.events.length === 0;
    document.getElementById("hist-empty").hidden = data.events.length !== 0;
  } catch (err) {
    console.error("history poll failed", err);
  }
}

document.getElementById("hist-reset-btn").addEventListener("click", async (e) => {
  if (!confirm("Clear all saved persons, the unique person counter, and event history?")) {
    return;
  }
  e.target.disabled = true;
  try {
    await fetch("/api/reset", { method: "POST" });
    knownEventIds = new Set();
    refreshHistory();
    refreshOccupancy();
    showToast("Event history cleared");
  } catch (err) {
    console.error("history reset failed", err);
    showToast("Reset failed — check connection");
  } finally {
    e.target.disabled = false;
  }
});

document.getElementById("reset-btn").addEventListener("click", async (e) => {
  if (!confirm("Clear all saved persons, the unique person counter, and event history?")) {
    return;
  }
  e.target.disabled = true;
  try {
    await fetch("/api/reset", { method: "POST" });
    knownEventIds = new Set();
    refreshOccupancy();
    refreshHistory();
    showToast("Saved persons and history cleared");
  } catch (err) {
    console.error("reset failed", err);
    showToast("Reset failed — check connection");
  } finally {
    e.target.disabled = false;
  }
});

// --- Live occupancy/zone-status feed (SSE) ------------------------------
// A single long-lived EventSource replaces 1s polling of /api/occupancy and
// /api/zone-status: the server pushes a named event only when that payload
// actually changes (see /api/stream in webapp.py). Opened once globally
// since it's push-based and cheap to leave connected; EventSource also
// auto-reconnects on drop.

const occupancyStream = new EventSource("/api/stream");
occupancyStream.addEventListener("occupancy", (e) => renderOccupancy(JSON.parse(e.data)));
occupancyStream.addEventListener("zone-status", (e) => renderZoneStatus(JSON.parse(e.data)));
occupancyStream.onerror = () => setConnected(false);
occupancyStream.onopen = () => setConnected(true);

// --- Per-tab lazy polling ----------------------------------------------
// History still polls on an interval only while its tab is active; live
// occupancy/zone-status come from the SSE stream above instead.

const tabPollers = {
  history: { refresh: refreshHistory, timer: null },
};

function stopAllPollers() {
  Object.values(tabPollers).forEach((p) => {
    if (p.timer) {
      clearInterval(p.timer);
      p.timer = null;
    }
  });
}

function onTabActivated(tabName) {
  stopAllPollers();
  const poller = tabPollers[tabName];
  if (!poller) return;
  poller.refresh();
  poller.timer = setInterval(poller.refresh, POLL_INTERVAL_MS);
}

onTabActivated("live");
refreshOccupancy();

// --- Start/Stop detector ----------------------------------------------

const startStopBtn = document.getElementById("start-stop-btn");
const detectorStatusEl = document.getElementById("detector-status");
let startStopPending = false;

function setStartStopUI(running) {
  if (startStopPending) return;
  startStopBtn.dataset.state = running ? "running" : "stopped";
  startStopBtn.textContent = running ? "Stop" : "Start";
  detectorStatusEl.textContent = running ? "Detector running" : "Detector stopped";
}

startStopBtn.addEventListener("click", async () => {
  const isRunning = startStopBtn.dataset.state === "running";
  const endpoint = isRunning ? "/api/stop" : "/api/start";
  startStopPending = true;
  startStopBtn.disabled = true;
  try {
    const res = await fetch(endpoint, { method: "POST" });
    const data = await res.json();
    startStopPending = false;
    setStartStopUI(data.running);
    showToast(data.running ? "Detector started" : "Detector stopped");
  } catch (err) {
    console.error("start/stop failed", err);
    startStopPending = false;
    showToast("Action failed — check connection");
  } finally {
    startStopBtn.disabled = false;
  }
});

// --- Zones tab -------------------------------------------------------

const zoneCanvas = document.getElementById("zone-canvas");
const zoneCtx = zoneCanvas.getContext("2d");
const zoneImg = document.getElementById("zone-snapshot");
const zoneNameInput = document.getElementById("zone-name");
const zoneTaskSelect = document.getElementById("zone-task-select");
const typeBtns = Array.from(document.querySelectorAll(".type-btn"));

let zoneMeta = { orig_w: 0, orig_h: 0, scale: 1 };
let zonePoints = []; // native-pixel [x, y] points for the in-progress polygon
let selectedType = "door";
let zoneSnapshotLoaded = false;
const CLOSE_RADIUS_PX = 10; // canvas CSS-pixel radius for closing the polygon

typeBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    typeBtns.forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    selectedType = btn.dataset.type;
  });
});

function nativeToCanvasCss(pt) {
  const cssScale = zoneCanvas.clientWidth / (zoneMeta.orig_w * zoneMeta.scale);
  return [pt[0] * zoneMeta.scale * cssScale, pt[1] * zoneMeta.scale * cssScale];
}

function canvasCssToNative(cssX, cssY) {
  const cssScale = zoneCanvas.clientWidth / (zoneMeta.orig_w * zoneMeta.scale);
  return [cssX / cssScale / zoneMeta.scale, cssY / cssScale / zoneMeta.scale];
}

function resizeZoneCanvas() {
  zoneCanvas.width = zoneCanvas.clientWidth;
  zoneCanvas.height = zoneCanvas.clientHeight;
  drawZoneOverlay();
}

function drawZoneOverlay() {
  zoneCtx.clearRect(0, 0, zoneCanvas.width, zoneCanvas.height);

  zoneCtx.strokeStyle = "#4da3ff";
  zoneCtx.fillStyle = "rgba(77, 163, 255, 0.2)";
  zoneCtx.lineWidth = 2;

  if (zonePoints.length > 0) {
    zoneCtx.beginPath();
    const start = nativeToCanvasCss(zonePoints[0]);
    zoneCtx.moveTo(start[0], start[1]);
    for (let i = 1; i < zonePoints.length; i++) {
      const p = nativeToCanvasCss(zonePoints[i]);
      zoneCtx.lineTo(p[0], p[1]);
    }
    zoneCtx.stroke();

    zonePoints.forEach((pt, i) => {
      const p = nativeToCanvasCss(pt);
      zoneCtx.beginPath();
      zoneCtx.arc(p[0], p[1], i === 0 ? 5 : 3, 0, Math.PI * 2);
      zoneCtx.fillStyle = i === 0 ? "#4da3ff" : "#e8eaf0";
      zoneCtx.fill();
    });
  }

  savedZones.forEach((zone) => {
    if (zone.points.length < 2) return;
    zoneCtx.beginPath();
    const start = nativeToCanvasCss(zone.points[0]);
    zoneCtx.moveTo(start[0], start[1]);
    for (let i = 1; i < zone.points.length; i++) {
      const p = nativeToCanvasCss(zone.points[i]);
      zoneCtx.lineTo(p[0], p[1]);
    }
    zoneCtx.closePath();
    zoneCtx.strokeStyle = "#4caf50";
    zoneCtx.fillStyle = "rgba(76, 175, 80, 0.12)";
    zoneCtx.stroke();
    zoneCtx.fill();
  });
}

function closeZonePolygon() {
  zoneCtx.fillStyle = "rgba(77, 163, 255, 0.2)";
  zoneCtx.beginPath();
  const start = nativeToCanvasCss(zonePoints[0]);
  zoneCtx.moveTo(start[0], start[1]);
  zonePoints.slice(1).forEach((pt) => {
    const p = nativeToCanvasCss(pt);
    zoneCtx.lineTo(p[0], p[1]);
  });
  zoneCtx.closePath();
  zoneCtx.fill();
}

zoneCanvas.addEventListener("click", (e) => {
  if (!zoneSnapshotLoaded) return;
  const rect = zoneCanvas.getBoundingClientRect();
  const cssX = e.clientX - rect.left;
  const cssY = e.clientY - rect.top;

  if (zonePoints.length >= 3) {
    const first = nativeToCanvasCss(zonePoints[0]);
    const dist = Math.hypot(cssX - first[0], cssY - first[1]);
    if (dist <= CLOSE_RADIUS_PX) {
      drawZoneOverlay();
      closeZonePolygon();
      return;
    }
  }

  zonePoints.push(canvasCssToNative(cssX, cssY));
  drawZoneOverlay();
});

document.getElementById("zone-undo-btn").addEventListener("click", () => {
  zonePoints.pop();
  drawZoneOverlay();
});

document.getElementById("zone-clear-btn").addEventListener("click", () => {
  zonePoints = [];
  drawZoneOverlay();
});

document.getElementById("zone-save-btn").addEventListener("click", async () => {
  const name = zoneNameInput.value.trim();
  if (!name) {
    showToast("Enter a zone name first");
    return;
  }
  if (zonePoints.length < 3) {
    showToast("Draw at least 3 points and close the shape");
    return;
  }
  try {
    await fetch("/api/zones", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name,
        type: selectedType,
        points: zonePoints,
        task: zoneTaskSelect.value,
      }),
    });
    zonePoints = [];
    zoneNameInput.value = "";
    zoneTaskSelect.value = "none";
    showToast("Zone saved");
    await refreshZones();
  } catch (err) {
    console.error("zone save failed", err);
    showToast("Save failed — check connection");
  }
});

let savedZones = [];

const TASK_LABELS = {
  none: "No task",
  alert_entry: "Alert on entry",
  alert_presence: "Alert on presence",
};

async function refreshZones() {
  try {
    const res = await fetch("/api/zones");
    savedZones = await res.json();

    const tbody = document.querySelector("#zone-table tbody");
    tbody.innerHTML = "";
    savedZones.forEach((zone, i) => {
      const tr = document.createElement("tr");
      tr.dataset.zoneId = zone.id;
      tr.style.animationDelay = `${i * 30}ms`;
      const nameTd = document.createElement("td");
      nameTd.textContent = zone.name;
      const typeTd = document.createElement("td");
      typeTd.textContent = zone.type;
      const ptsTd = document.createElement("td");
      ptsTd.textContent = `${zone.points.length} pts`;

      const taskTd = document.createElement("td");
      const taskSelect = document.createElement("select");
      taskSelect.className = "zone-task-edit";
      Object.entries(TASK_LABELS).forEach(([value, label]) => {
        const opt = document.createElement("option");
        opt.value = value;
        opt.textContent = label;
        taskSelect.appendChild(opt);
      });
      taskSelect.value = zone.task || "none";
      taskSelect.addEventListener("change", async () => {
        taskSelect.disabled = true;
        try {
          await fetch(`/api/zones/${zone.id}`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ task: taskSelect.value }),
          });
          showToast("Zone task updated");
          await refreshZones();
        } catch (err) {
          console.error("zone task update failed", err);
          showToast("Update failed — check connection");
        } finally {
          taskSelect.disabled = false;
        }
      });
      taskTd.appendChild(taskSelect);

      const alertTd = document.createElement("td");
      alertTd.className = "zone-alert-cell";
      alertTd.textContent = "-";

      const actionTd = document.createElement("td");
      const delBtn = document.createElement("button");
      delBtn.textContent = "Delete";
      delBtn.className = "reset-btn";
      delBtn.addEventListener("click", async () => {
        if (!confirm(`Delete zone "${zone.name}"?`)) return;
        delBtn.disabled = true;
        try {
          await fetch(`/api/zones/${zone.id}`, { method: "DELETE" });
          showToast("Zone deleted");
          await refreshZones();
        } catch (err) {
          console.error("zone delete failed", err);
          showToast("Delete failed — check connection");
          delBtn.disabled = false;
        }
      });
      actionTd.appendChild(delBtn);
      tr.append(nameTd, typeTd, ptsTd, taskTd, alertTd, actionTd);
      tbody.appendChild(tr);
    });
    document.getElementById("zone-table").hidden = savedZones.length === 0;
    document.getElementById("zone-empty").hidden = savedZones.length !== 0;
    drawZoneOverlay();
    refreshZoneStatus();
  } catch (err) {
    console.error("zones poll failed", err);
  }
}

function renderZoneStatus(data) {
  data.zones.forEach((zs) => {
    const row = document.querySelector(`#zone-table tr[data-zone-id="${zs.id}"]`);
    if (!row) return;
    const alertCell = row.querySelector(".zone-alert-cell");
    if (!alertCell) return;
    if (zs.task === "none") {
      alertCell.textContent = "-";
      alertCell.classList.remove("zone-alert-active");
    } else {
      alertCell.textContent = zs.alert ? "ALERT" : "clear";
      alertCell.classList.toggle("zone-alert-active", zs.alert);
    }
    row.classList.toggle("zone-row-alert", zs.alert);
  });
}

async function refreshZoneStatus() {
  try {
    const res = await fetch("/api/zone-status");
    renderZoneStatus(await res.json());
  } catch (err) {
    console.error("zone-status poll failed", err);
  }
}

async function loadZoneSnapshot() {
  try {
    const [metaRes] = await Promise.all([fetch("/api/zone-snapshot/meta")]);
    zoneMeta = await metaRes.json();
    zoneImg.src = `/api/zone-snapshot?t=${Date.now()}`;
  } catch (err) {
    console.error("zone snapshot load failed", err);
    showToast("Could not load snapshot — is the detector running?");
  }
}

zoneImg.addEventListener("load", () => {
  zoneSnapshotLoaded = true;
  resizeZoneCanvas();
});

window.addEventListener("resize", () => {
  if (zoneSnapshotLoaded) resizeZoneCanvas();
});

document.querySelector('.tab-btn[data-tab="zones"]').addEventListener("click", () => {
  loadZoneSnapshot();
  refreshZones();
  refreshZoneStatus();
});

// --- People tab --------------------------------------------------------

const peopleCanvas = document.getElementById("people-canvas");
const peopleCtx = peopleCanvas.getContext("2d");
const peopleImg = document.getElementById("people-snapshot");
const peopleNameForm = document.getElementById("people-name-form");
const peopleSelectedLabel = document.getElementById("people-selected-label");
const peopleNameInput = document.getElementById("people-name-input");

let peopleMeta = { orig_w: 0, orig_h: 0, scale: 1 };
let peopleSnapshotLoaded = false;
let livePeople = []; // [{person_id, box:[x1,y1,x2,y2], name}]
let selectedPersonId = null;
let peopleLiveTimer = null;

const TRACK_COLOR_STORAGE_KEY = "trackBorderColor";
const trackColorInput = document.getElementById("track-color-input");
let trackBorderColor = localStorage.getItem(TRACK_COLOR_STORAGE_KEY) || "#4caf50";
trackColorInput.value = trackBorderColor;

fetch("/api/box-color")
  .then((res) => (res.ok ? res.json() : null))
  .then((data) => {
    if (data && data.color) {
      trackBorderColor = data.color;
      trackColorInput.value = trackBorderColor;
      localStorage.setItem(TRACK_COLOR_STORAGE_KEY, trackBorderColor);
      drawPeopleOverlay();
    }
  })
  .catch(() => {});

trackColorInput.addEventListener("input", () => {
  trackBorderColor = trackColorInput.value;
  localStorage.setItem(TRACK_COLOR_STORAGE_KEY, trackBorderColor);
  drawPeopleOverlay();
  fetch("/api/box-color", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ color: trackBorderColor }),
  }).catch(() => {});
});

function peopleNativeToCanvasCss(x, y) {
  const cssScale = peopleCanvas.clientWidth / (peopleMeta.orig_w * peopleMeta.scale);
  return [x * peopleMeta.scale * cssScale, y * peopleMeta.scale * cssScale];
}

function peopleCanvasCssToNative(cssX, cssY) {
  const cssScale = peopleCanvas.clientWidth / (peopleMeta.orig_w * peopleMeta.scale);
  return [cssX / cssScale / peopleMeta.scale, cssY / cssScale / peopleMeta.scale];
}

function resizePeopleCanvas() {
  peopleCanvas.width = peopleCanvas.clientWidth;
  peopleCanvas.height = peopleCanvas.clientHeight;
  drawPeopleOverlay();
}

function drawPeopleOverlay() {
  peopleCtx.clearRect(0, 0, peopleCanvas.width, peopleCanvas.height);
  peopleCtx.lineWidth = 2;
  peopleCtx.font = "13px sans-serif";

  livePeople.forEach((p) => {
    const [x1, y1] = peopleNativeToCanvasCss(p.box[0], p.box[1]);
    const [x2, y2] = peopleNativeToCanvasCss(p.box[2], p.box[3]);
    const selected = p.person_id === selectedPersonId;
    peopleCtx.strokeStyle = selected ? "#4da3ff" : trackBorderColor;
    peopleCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    const label = p.name || `Person-${p.person_id}`;
    const textWidth = peopleCtx.measureText(label).width;
    peopleCtx.fillStyle = selected ? "#4da3ff" : trackBorderColor;
    peopleCtx.fillRect(x1, Math.max(0, y1 - 18), textWidth + 8, 18);
    peopleCtx.fillStyle = "#0b0d12";
    peopleCtx.fillText(label, x1 + 4, Math.max(12, y1 - 5));
  });
}

peopleCanvas.addEventListener("click", (e) => {
  if (!peopleSnapshotLoaded) return;
  const rect = peopleCanvas.getBoundingClientRect();
  const [nx, ny] = peopleCanvasCssToNative(e.clientX - rect.left, e.clientY - rect.top);

  const hit = livePeople.find((p) => {
    const [x1, y1, x2, y2] = p.box;
    return nx >= x1 && nx <= x2 && ny >= y1 && ny <= y2;
  });

  if (!hit) {
    selectedPersonId = null;
    peopleNameForm.hidden = true;
    drawPeopleOverlay();
    return;
  }

  selectedPersonId = hit.person_id;
  peopleSelectedLabel.textContent = `Person-${hit.person_id}`;
  peopleNameInput.value = hit.name || "";
  peopleNameForm.hidden = false;
  peopleNameInput.focus();
  drawPeopleOverlay();
});

document.getElementById("people-cancel-btn").addEventListener("click", () => {
  selectedPersonId = null;
  peopleNameForm.hidden = true;
  drawPeopleOverlay();
});

document.getElementById("people-save-btn").addEventListener("click", async () => {
  if (selectedPersonId === null) return;
  const name = peopleNameInput.value.trim();
  if (!name) {
    showToast("Enter a name first");
    return;
  }
  try {
    await fetch(`/api/people/${selectedPersonId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    showToast("Name saved");
    peopleNameForm.hidden = true;
    selectedPersonId = null;
    await Promise.all([refreshLiveTracks(), refreshPeopleTable()]);
  } catch (err) {
    console.error("people save failed", err);
    showToast("Save failed — check connection");
  }
});

async function refreshPeopleTable() {
  try {
    const res = await fetch("/api/people");
    const data = await res.json();

    const tbody = document.querySelector("#people-table tbody");
    tbody.innerHTML = "";
    data.people.forEach((p, i) => {
      const tr = document.createElement("tr");
      tr.style.animationDelay = `${i * 30}ms`;
      const idTd = document.createElement("td");
      idTd.textContent = `Person-${p.person_id}`;
      const nameTd = document.createElement("td");
      nameTd.textContent = p.name;
      const actionTd = document.createElement("td");
      const delBtn = document.createElement("button");
      delBtn.textContent = "Delete";
      delBtn.className = "reset-btn";
      delBtn.addEventListener("click", async () => {
        delBtn.disabled = true;
        try {
          await fetch(`/api/people/${p.person_id}`, { method: "DELETE" });
          showToast("Name removed");
          await Promise.all([refreshLiveTracks(), refreshPeopleTable()]);
        } catch (err) {
          console.error("people delete failed", err);
          showToast("Delete failed — check connection");
          delBtn.disabled = false;
        }
      });
      actionTd.appendChild(delBtn);
      tr.append(idTd, nameTd, actionTd);
      tbody.appendChild(tr);
    });
    document.getElementById("people-table").hidden = data.people.length === 0;
    document.getElementById("people-empty").hidden = data.people.length !== 0;
  } catch (err) {
    console.error("people table poll failed", err);
  }
}

async function refreshLiveTracks() {
  try {
    const res = await fetch("/api/live-tracks");
    const data = await res.json();
    livePeople = data.people;
    drawPeopleOverlay();
  } catch (err) {
    console.error("live-tracks poll failed", err);
  }
}

async function loadPeopleSnapshot() {
  try {
    const metaRes = await fetch("/api/zone-snapshot/meta");
    peopleMeta = await metaRes.json();
    peopleImg.src = `/api/zone-snapshot?t=${Date.now()}`;
  } catch (err) {
    console.error("people snapshot load failed", err);
    showToast("Could not load snapshot — is the detector running?");
  }
}

peopleImg.addEventListener("load", () => {
  peopleSnapshotLoaded = true;
  resizePeopleCanvas();
});

window.addEventListener("resize", () => {
  if (peopleSnapshotLoaded) resizePeopleCanvas();
});

function startPeopleLivePolling() {
  stopPeopleLivePolling();
  peopleLiveTimer = setInterval(() => {
    loadPeopleSnapshot();
    refreshLiveTracks();
  }, POLL_INTERVAL_MS);
}

function stopPeopleLivePolling() {
  if (peopleLiveTimer) {
    clearInterval(peopleLiveTimer);
    peopleLiveTimer = null;
  }
}

document.querySelector('.tab-btn[data-tab="people"]').addEventListener("click", () => {
  loadPeopleSnapshot();
  refreshLiveTracks();
  refreshPeopleTable();
  startPeopleLivePolling();
});

tabBtns.forEach((btn) => {
  if (btn.dataset.tab !== "people") {
    btn.addEventListener("click", stopPeopleLivePolling);
  }
});
