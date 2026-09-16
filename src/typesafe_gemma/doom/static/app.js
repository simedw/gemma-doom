const $ = (id) => document.getElementById(id);
const fieldNames = ["move", "strafe", "turn", "fire", "use"];
const keyNames = new Set([
  "KeyW",
  "KeyS",
  "KeyA",
  "KeyD",
  "ArrowLeft",
  "ArrowRight",
  "Space",
  "KeyE",
]);
let socket,
  state = {},
  connected = false,
  pressed = new Set(),
  latestBlob = null,
  drawing = false;
let receivedFrameId = 0,
  lastDrawnFrame = 0;
const canvas = $("game"),
  ctx = canvas.getContext("2d");
ctx.imageSmoothingEnabled = false;
$("actions").innerHTML = fieldNames
  .map(
    (name) =>
      `<div class="action-row" id="action-${name}"><span class="action-name">${name}</span><div class="action-bar"><div class="action-fill"></div></div><span class="action-value">—</span></div>`,
  )
  .join("");

function send(command) {
  if (socket?.readyState === WebSocket.OPEN)
    socket.send(JSON.stringify(command));
}
function clearKeys() {
  pressed.clear();
  if (state.is_controller) send({ type: "keys", keys: [] });
}
function toast(message) {
  $("toast").textContent = message;
  $("toast").hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => ($("toast").hidden = true), 4000);
}
function focusGame() {
  $("viewport").focus({ preventScroll: true });
}

async function drawLatest() {
  if (drawing) return;
  drawing = true;
  while (latestBlob) {
    const { blob, frameId } = latestBlob;
    latestBlob = null;
    try {
      const bitmap = await createImageBitmap(blob);
      ctx.drawImage(bitmap, 0, 0, 640, 480);
      bitmap.close();
      lastDrawnFrame = frameId;
      $("frame-label").textContent = `FRAME ${frameId}`;
    } catch (error) {
      console.error("Frame decode failed", error);
    }
  }
  drawing = false;
}

function render(next) {
  state = next;
  receivedFrameId = next.frame_id ?? receivedFrameId;
  const modelReady = next.model?.status === "ready";
  const owner = next.is_controller;
  const finished = next.status === "finished";
  $("connection").textContent = owner ? "Connected" : "Spectating";
  $("health").textContent = next.health ?? "—";
  $("health").style.color = next.health < 30 ? "#f0aa8b" : "";
  for (const name of ["ammo", "kills", "tic"])
    $(name).textContent = next[name] ?? "—";
  $("episode").textContent = String(next.episode ?? 1).padStart(2, "0");
  $("scenario").value = next.scenario ?? "defend_the_center";
  $("scenario").disabled = !owner;
  $("play-state").textContent = finished
    ? "FINISHED"
    : next.pending_step
      ? "MODEL STEP"
      : next.paused
        ? "PAUSED"
        : next.mode === "autopilot"
          ? "AI PLAYING"
          : "LIVE";
  $("play").textContent = next.paused ? "▶ Play" : "Ⅱ Pause";
  $("play").disabled =
    !owner ||
    next.status !== "ready" ||
    (next.mode === "autopilot" && !modelReady);
  $("reset").disabled = !owner || next.status === "starting";
  $("step").disabled =
    !owner || !modelReady || next.model?.busy || next.pending_step || finished;
  $("takeover").textContent = owner ? "Take control" : "Claim controls";
  $("takeover").disabled = !connected;
  $("model-status").textContent = modelReady
    ? `GPU ${next.model.gpu} · ready`
    : next.model?.status === "loading"
      ? "Loading weights…"
      : ["compiling", "warming"].includes(next.model?.status)
        ? "Preparing model…"
        : next.model?.status === "disabled"
          ? "Manual demo · model off"
          : "Model unavailable";
  $("model-status").classList.toggle(
    "error-text",
    next.model?.status === "error",
  );
  $("thinking").textContent = next.model?.busy
    ? "THINKING"
    : next.mode === "manual"
      ? "STANDBY"
      : modelReady
        ? "READY"
        : "LOADING";
  $("thinking").classList.toggle("busy", Boolean(next.model?.busy));
  for (const button of document.querySelectorAll("[data-mode]")) {
    button.classList.toggle("active", button.dataset.mode === next.mode);
    button.disabled =
      !owner || (button.dataset.mode !== "manual" && !modelReady);
  }
  $("mode-description").textContent = {
    manual: "Your keyboard controls the game. Gemma is on standby.",
    copilot: "You play. Gemma watches the same frames and suggests what to do.",
    autopilot: "Gemma controls the game. Take control at any time.",
  }[next.mode];
  const prediction = next.prediction;
  $("latency").textContent = prediction
    ? Math.round(prediction.timing_ms.total)
    : "—";
  $("decision-frame").textContent = prediction
    ? `FRAME ${prediction.frame_id} · TICK ${prediction.tic} · ${(prediction.since_ms / 1000).toFixed(1)}s ago`
    : "No observation yet";
  $("json").textContent = prediction
    ? JSON.stringify(prediction.result, null, 2)
    : "Waiting for the first observation.";
  for (const name of fieldNames) {
    const row = $(`action-${name}`),
      value = prediction?.result[name];
    row.querySelector(".action-value").textContent =
      value === undefined ? "—" : String(value);
    row.classList.toggle(
      "on",
      value !== undefined && value !== false && value !== "none",
    );
    const p = prediction?.scores[name]?.label_probabilities[String(value)] ?? 0;
    row.querySelector(".action-fill").style.width = `${p * 100}%`;
    row.title = prediction
      ? Object.entries(prediction.scores[name].label_probabilities)
          .map(([v, p]) => `${v}: ${(p * 100).toFixed(1)}%`)
          .join(" · ")
      : "";
  }
  const action = next.applied ?? {};
  const active = [];
  if (action.move && action.move !== "none")
    active.push(action.move.toUpperCase());
  if (action.strafe && action.strafe !== "none")
    active.push(`STRAFE ${action.strafe.toUpperCase()}`);
  if (action.turn && action.turn !== "none")
    active.push(`TURN ${action.turn.toUpperCase()}`);
  if (action.fire) active.push("FIRE");
  if (action.use) active.push("USE");
  $("applied").replaceChildren(
    ...(active.length
      ? active.map((text) => {
          const span = document.createElement("span");
          span.className = "key-active";
          span.textContent = text;
          return span;
        })
      : [document.createTextNode("No buttons pressed")]),
  );
  const overlayVisible =
    (next.paused && !next.pending_step && !prediction && next.tic <= 14) ||
    finished ||
    next.status === "starting" ||
    next.status === "error";
  $("overlay").hidden = !overlayVisible;
  $("overlay-title").textContent =
    next.status === "error"
      ? "Arena unavailable."
      : finished
        ? "End of this run."
        : next.status === "starting"
          ? "Opening the arena…"
          : prediction
            ? "Take a closer look."
            : "Ready when you are.";
  $("overlay-description").textContent =
    next.error ||
    (finished
      ? "Reset for another run, or try a different arena."
      : prediction
        ? "Inspect the decision, take another model step, or press Play."
        : "Press Play to enter the arena, or inspect one model step.");
  $("focus-hint").hidden =
    next.paused ||
    next.mode === "autopilot" ||
    document.activeElement === $("viewport") ||
    !owner;
  $("event").textContent =
    next.error || next.model?.error || next.event || "Starting…";
  $("event").classList.toggle(
    "error-text",
    Boolean(next.error || next.model?.error),
  );
  if (document.activeElement !== $("hz")) $("hz").value = next.model_hz ?? 5;
  $("hz-value").textContent = `${$("hz").value} / sec`;
  if (document.activeElement !== $("tics"))
    $("tics").value = next.step_tics ?? 4;
  $("hz").disabled = $("tics").disabled = !owner;
}

function connect() {
  socket = new WebSocket(
    `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/ws`,
  );
  socket.binaryType = "blob";
  socket.onopen = () => {
    connected = true;
    $("connection-dot").classList.add("connected");
  };
  socket.onmessage = (event) => {
    if (typeof event.data === "string") {
      const data = JSON.parse(event.data);
      if (data.type === "state") render(data);
      else if (data.type === "error") toast(data.message);
    } else {
      latestBlob = { blob: event.data, frameId: receivedFrameId };
      drawLatest();
    }
  };
  socket.onclose = () => {
    connected = false;
    pressed.clear();
    $("connection-dot").classList.remove("connected");
    $("connection").textContent = "Reconnecting…";
    document
      .querySelectorAll("button,select,input")
      .forEach((el) => (el.disabled = true));
    setTimeout(connect, 1500);
  };
  socket.onerror = () => socket.close();
}

$("play").onclick = () => {
  clearKeys();
  send({ type: "pause", value: !state.paused });
  focusGame();
};
$("reset").onclick = () => {
  clearKeys();
  send({ type: "reset" });
};
$("step").onclick = () => {
  clearKeys();
  send({ type: "step" });
};
$("takeover").onclick = () => {
  clearKeys();
  send({ type: state.is_controller ? "takeover" : "claim" });
  focusGame();
};
$("scenario").onchange = (event) => {
  clearKeys();
  send({ type: "scenario", value: event.target.value });
};
document.querySelectorAll("[data-mode]").forEach(
  (button) =>
    (button.onclick = () => {
      clearKeys();
      send({ type: "mode", value: button.dataset.mode });
      focusGame();
    }),
);
function settings() {
  send({
    type: "settings",
    hz: Number($("hz").value),
    tics: Number($("tics").value),
  });
}
$("hz").oninput = () => ($("hz-value").textContent = `${$("hz").value} / sec`);
$("hz").onchange = settings;
$("tics").onchange = settings;
$("viewport").onclick = focusGame;
$("viewport").onfocus = () => ($("focus-hint").hidden = true);
$("viewport").onblur = clearKeys;
window.addEventListener("keydown", (event) => {
  if (event.code === "Escape" && state.is_controller) {
    event.preventDefault();
    clearKeys();
    send({ type: "pause", value: true });
    return;
  }
  if (document.activeElement !== $("viewport") || !keyNames.has(event.code))
    return;
  event.preventDefault();
  if (!state.is_controller || state.paused || state.mode === "autopilot")
    return;
  pressed.add(event.code);
  send({ type: "keys", keys: [...pressed] });
});
window.addEventListener("keyup", (event) => {
  if (pressed.delete(event.code)) {
    event.preventDefault();
    send({ type: "keys", keys: [...pressed] });
  }
});
window.addEventListener("blur", clearKeys);
document.addEventListener("visibilitychange", () => {
  if (document.hidden) clearKeys();
});
setInterval(() => {
  if (pressed.size && connected && state.is_controller)
    send({ type: "keys", keys: [...pressed] });
}, 100);
connect();
