"use strict";

const $ = (id) => document.getElementById(id);
const STORE = "hu20-local-play-v1";
let saved = JSON.parse(localStorage.getItem(STORE) || "{}");
let state = null;
let busy = false;
let processingBot = false;
let availableModels = [];

function persist() { localStorage.setItem(STORE, JSON.stringify(saved)); }
function bb(chips) { return `${(chips / 100).toFixed(2).replace(/\.00$/, "")} BB`; }
function signed(chips) { return `${chips >= 0 ? "+" : "−"}${bb(Math.abs(chips))}`; }
function node(tag, className, content) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (content !== undefined) element.textContent = content;
  return element;
}
function clear(element) { element.replaceChildren(); }
function notice(message, error = false) { $("notice").textContent = message; $("notice").classList.toggle("error", error); $("retry").hidden = !error; }
function connection(message, good = false) { $("connection").textContent = message; $("connection").classList.toggle("online", good); }
function show(id) { for (const name of ["gate", "setup", "game"]) $(name).hidden = name !== id; }

async function request(path, { method = "GET", body, key } = {}) {
  const headers = { "X-Play-Token": saved.token || "" };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (key) headers["Idempotency-Key"] = key;
  const response = await fetch(path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body), cache: "no-store" });
  const result = await response.json();
  if (!response.ok) {
    const error = new Error(result.error || `Request failed (${response.status})`);
    error.status = response.status;
    throw error;
  }
  connection("Connected to local table", true);
  return result;
}

async function mutate(path, body) {
  if (busy) return;
  busy = true; render();
  const operation = saved.pending || { path, body, key: crypto.randomUUID() };
  saved.pending = operation; persist();
  try {
    const response = await request(operation.path, { method: "POST", body: operation.body, key: operation.key });
    saved.pending = null;
    saved.sessionId = response.sessionId;
    persist();
    state = response;
    notice("");
    return response;
  } catch (error) {
    if (error.status >= 400 && error.status < 500) { saved.pending = null; persist(); }
    if (error.status === 403) { saved.token = ""; persist(); }
    connection("Disconnected or request failed");
    notice(`${error.message}. ${saved.pending ? "Retry the same operation" : "Refresh table state"}; no action will be chosen for you.`, true);
    throw error;
  } finally {
    busy = false; render();
  }
}

async function recover() {
  if (!saved.token) { show("gate"); connection("Access token needed"); return; }
  if (saved.pending) {
    const pending = saved.pending;
    try { await mutate(pending.path, pending.body); } catch (_) { return; }
  }
  if (!saved.sessionId) {
    show("setup");
    try {
      const catalog = await request("/api/models");
      availableModels = catalog.models;
      clear($("model-version"));
      for (const item of availableModels) {
        const option = node("option", "", item.name);
        option.value = item.version; $("model-version").append(option);
      }
      $("model-version").value = catalog.default || "";
      $("model-version").hidden = $("model-version-label").hidden = !availableModels.length;
      const model = availableModels.find(item => item.version === catalog.default) || await request("/api/model");
      $("setup-model").textContent = `${model.name} · HU20 · SHA-256 ${model.sha256}`;
      $("setup-title").textContent = model.adapter === "uniform-restricted-v1"
        ? "Uniform-random calibration" : `Play ${model.name.split(" · ")[0]}`;
      if (model.benchmarkOnly) {
        for (const input of document.querySelectorAll('input[name="sessionType"]')) {
          input.checked = input.value === "benchmark"; input.disabled = input.value !== "benchmark";
        }
        for (const input of document.querySelectorAll('input[name="playMode"]')) {
          input.checked = input.value === "restricted"; input.disabled = input.value !== "restricted";
        }
        updateSetup();
      }
    } catch (error) {
      connection("Disconnected"); notice(error.message, true);
      if (error.status === 403) { saved.token = ""; persist(); show("gate"); }
    }
    return;
  }
  try {
    state = await request(`/api/sessions/${saved.sessionId}`);
    render();
    await maybeAdvanceBot();
  } catch (error) {
    connection("Disconnected");
    notice(error.message, true);
    if (error.status === 403) { saved.token = ""; persist(); show("gate"); return; }
    if (error.status === 404) { saved.sessionId = null; persist(); await recover(); }
  }
}

function card(value, hidden = false) {
  const c = node("span", `card ${hidden ? "hidden-card" : (value?.endsWith("h") || value?.endsWith("d") ? "red" : "")}`, hidden ? "◆" : `${value[0]}${{ c:"♣", d:"♦", h:"♥", s:"♠" }[value[1]]}`);
  c.setAttribute("aria-label", hidden ? "Hidden card" : value);
  return c;
}
function cards(target, values, backCount = 0) {
  clear(target);
  for (const value of values) target.append(card(value));
  for (let i = 0; i < backCount; i++) target.append(card("", true));
}
function seat(target, player, isHuman, hand) {
  clear(target);
  target.classList.toggle("acting", hand.actor === player.seat);
  const head = node("div", "seat-head");
  head.append(node("strong", "", isHuman ? "You" : state.model.name.split(" · ")[0]));
  if (hand.button === player.seat) head.append(node("span", "badge", "D · SB"));
  else head.append(node("span", "badge", "BB"));
  target.append(head);
  target.append(node("div", "stack", bb(player.stack)));
  const visible = isHuman ? hand.humanCards : player.shownCards;
  const deck = node("div", "cards seat-cards");
  cards(deck, visible, isHuman || visible.length ? 0 : 2);
  target.append(deck);
  const line = [player.folded ? "Folded" : player.allIn ? "All-in" : "", player.streetBet ? `Street ${bb(player.streetBet)}` : "", player.contributed ? `In pot ${bb(player.contributed)}` : ""].filter(Boolean).join(" · ");
  target.append(node("small", "seat-detail", line || "Waiting"));
}
function button(label, action, emphasis = false) {
  const control = node("button", emphasis ? "primary" : "action-button", label);
  control.type = "button";
  control.disabled = busy || !!saved.pending;
  control.addEventListener("click", action);
  return control;
}
async function act(kind, raiseTo = null) {
  if (!state?.hand) return;
  try {
    await mutate(`/api/sessions/${state.sessionId}/actions`, { handId: state.hand.id, revision: state.revision, kind, raiseTo });
    await maybeAdvanceBot();
  } catch (_) { /* The pending operation is retained for an exact retry. */ }
}
function exactChips(raw) {
  if (!/^\d+(?:\.\d{1,2})?$/.test(raw.trim())) return null;
  const [whole, fraction = ""] = raw.trim().split(".");
  const chips = Number(whole) * 100 + Number(fraction.padEnd(2, "0"));
  return Number.isSafeInteger(chips) ? chips : null;
}
function renderControls(hand) {
  const target = $("controls"); clear(target);
  if (!hand.legal) return;
  if (state.playMode === "restricted") {
    const group = node("div", "button-grid");
    for (const item of hand.menu) {
      const label = item.kind === "raise" ? `${item.label} · ${bb(item.raiseTo)}` : item.kind === "call" ? `Call ${bb(hand.legal.call)}` : item.kind;
      group.append(button(label, () => act(item.kind, item.raiseTo), item.kind === "check" || item.kind === "call"));
    }
    target.append(group);
    return;
  }
  const base = node("div", "button-grid");
  for (const kind of ["fold", "check", "call"]) if (hand.legal.kinds.includes(kind)) {
    base.append(button(kind === "call" ? `Call ${bb(hand.legal.call)}` : kind, () => act(kind), kind !== "fold"));
  }
  target.append(base);
  if (!hand.legal.kinds.includes("raise")) return;
  const panel = node("div", "raise-panel");
  panel.append(node("div", "raise-title", `Raise to · ${bb(hand.legal.minRaiseTo)} min / ${bb(hand.legal.maxRaiseTo)} max`));
  const row = node("div", "raise-row");
  const slider = node("input"); slider.type = "range"; slider.min = hand.legal.minRaiseTo; slider.max = hand.legal.maxRaiseTo; slider.step = "1"; slider.value = hand.legal.minRaiseTo; slider.disabled = busy;
  slider.setAttribute("aria-label", "Exact raise-to amount in chips");
  const field = node("input"); field.type = "text"; field.inputMode = "decimal"; field.value = (hand.legal.minRaiseTo / 100).toFixed(2); field.disabled = busy;
  field.setAttribute("aria-label", "Exact raise-to amount in BB");
  slider.addEventListener("input", () => { field.value = (Number(slider.value) / 100).toFixed(2); });
  field.addEventListener("input", () => { const value = exactChips(field.value); if (value !== null && value >= Number(slider.min) && value <= Number(slider.max)) slider.value = value; });
  const submit = button("Raise", () => {
    const chips = exactChips(field.value);
    if (chips === null || chips < hand.legal.minRaiseTo || chips > hand.legal.maxRaiseTo) { notice("Enter an exact chip amount within the displayed legal bounds.", true); return; }
    act("raise", chips);
  }, true);
  row.append(slider, field, node("span", "unit", "BB"), submit); panel.append(row);
  const presets = node("div", "presets");
  for (const preset of hand.presets) {
    const item = button(`${preset.label} · ${bb(preset.raiseTo)}`, () => { field.value = (preset.raiseTo / 100).toFixed(2); slider.value = preset.raiseTo; }, false);
    item.disabled = busy || !preset.available;
    presets.append(item);
  }
  panel.append(presets); target.append(panel);
}
function eventText(event) {
  if (event.event === "blind") return `${event.seat === 0 ? "You" : "Bot"} posts ${bb(event.amount)}`;
  if (event.event === "action") return `${event.seat === 0 ? "You" : "Bot"} ${event.kind === "raise" ? `raise to ${bb(event.raiseTo)}` : event.kind === "call" ? `call ${bb(event.paid)}` : event.kind}`;
  if (event.event === "board") return `${event.street}: ${event.cards.join(" ")}`;
  if (event.event === "shown") return `${event.seat === 0 ? "You" : "Bot"} show ${event.cards.join(" ")}`;
  if (event.event === "mucked") return `${event.seat === 0 ? "You" : "Bot"} muck`;
  return "";
}
async function loadDiagnostics() {
  if (!state || state.visibility !== "developer" || state.phase !== "finished") return;
  try {
    const result = await request(`/api/sessions/${state.sessionId}/hands/${state.hand.id}/diagnostics`);
    $("diagnostics").textContent = `After-hand lookup: ${result.trained} trained · ${result.fallback} fallback`;
  } catch (_) { /* Diagnostics do not affect play. */ }
}
function benchmarkLine(label, value) {
  const row = node("div", "benchmark-line");
  row.append(node("span", "", label), node("strong", "", value));
  return row;
}
function renderBenchmarkResult(report) {
  const panel = $("benchmark-results");
  panel.hidden = !report;
  if (!report) return;
  $("benchmark-status").textContent = report.status === "COMPLETE" ? "COMPLETE" : "INCOMPLETE · ENDED EARLY";
  const summary = $("benchmark-summary"); clear(summary);
  const lines = [
    ["Benchmark ID", report.benchmarkId], ["Protocol", report.protocolVersion],
    ["Model", `${report.model.name} · SHA-256 ${report.model.sha256}`],
    ["Game / schema", `${report.game} / ${report.schema}`],
    ["Mode / adapter", `${report.playMode} / ${report.adapter}`],
    ["Hands", `${report.completedHands} / ${report.targetHands}`],
    ["Human net", `${report.netChips} chips · ${signed(report.netChips)}`],
    ["Raw BB/100", report.bbPer100 === null ? "No completed hands" : report.bbPer100.toFixed(2)],
    ["Average pot", report.averagePotBB === null ? "No completed hands" : `${report.averagePotChips.toFixed(1)} chips · ${report.averagePotBB.toFixed(2)} BB`],
    ["Button / SB", `${report.buttonSB.hands} hands · ${signed(report.buttonSB.netChips)}`],
    ["Big blind", `${report.bigBlind.hands} hands · ${signed(report.bigBlind.netChips)}`],
    ["Wins / losses / ties", `${report.wins} / ${report.losses} / ${report.ties}`],
    ["Started", report.startedAt], ["Ended", report.endedAt],
    ["Source / interface", `${report.sourceVersion} / ${report.interfaceVersion}`]
  ];
  if (report.fallbackSummary) {
    const f = report.fallbackSummary;
    lines.push(["Bot trained / fallback lookups", `${f.trainedLookups} / ${f.fallbackLookups}`]);
    lines.push(["Fallback share", f.fallbackPercent === null ? "No bot lookups" : `${f.fallbackPercent.toFixed(2)}%`]);
    lines.push(["Hands with fallback", `${f.handsWithFallback} / ${report.completedHands} (${f.handsWithFallbackFraction === null ? "n/a" : `${(100 * f.handsWithFallbackFraction).toFixed(2)}%`})`]);
  }
  for (const [label, value] of lines) summary.append(benchmarkLine(label, value));
}
function render() {
  if (!saved.token) { show("gate"); return; }
  if (!state) { show("setup"); return; }
  show("game");
  const isBenchmark = state.sessionType === "benchmark";
  const activeBenchmark = isBenchmark && state.benchmark.status === "ACTIVE";
  $("mode").textContent = `${isBenchmark ? "HUMAN BENCHMARK / " : ""}${state.playMode === "free" ? "FREE SIZING · EXPERIMENTAL" : "RESTRICTED RESEARCH"} / ${state.visibility === "benchmark" ? "BENCHMARK-SAFE" : "DEVELOPER"}`;
  $("model").textContent = `${state.model.name} · ${state.model.sha256.slice(0, 12)}…`;
  $("session-stat").hidden = activeBenchmark;
  if (!activeBenchmark) $("session-bb").textContent = signed(state.sessionChips);
  $("progress").hidden = !isBenchmark;
  if (isBenchmark) $("progress-text").textContent = activeBenchmark && state.hand
    ? `Hand ${state.hand.number + 1} / ${state.benchmark.targetHands}`
    : `${state.benchmark.completedHands} / ${state.benchmark.targetHands} completed`;
  $("model-details").textContent = `Session ${state.sessionId}${isBenchmark ? ` · Benchmark ${state.benchmark.id}` : ""} · ${state.model.game} · ${state.model.schema} · ${state.model.format} · SHA-256 ${state.model.sha256} · ${state.model.adapter}`;
  $("benchmark-end").hidden = !activeBenchmark;
  $("benchmark-end").disabled = busy || !!saved.pending;
  renderBenchmarkResult(state.benchmarkResult || null);
  $("diagnostics").textContent = "";
  if (!state.hand) {
    $("turn").textContent = state.phase === "aborted" ? "Benchmark ended early" : "Ready to deal";
    $("result").textContent = state.phase === "aborted" ? "Completed hands are retained in the result." : `Start a 20 BB hand against ${state.model.name.split(" · ")[0]}.`;
    $("new-hand").hidden = state.phase === "aborted"; $("new-hand").disabled = busy || !!saved.pending;
    $("new-hand").textContent = activeBenchmark ? `Deal hand 1 / ${state.benchmark.targetHands}` : "Deal next hand";
    clear($("controls")); clear($("events")); clear($("board")); clear($("bot-seat")); clear($("human-seat"));
    $("street").textContent = "PRE-FLOP"; $("pot").textContent = "POT · 0 BB";
    return;
  }
  const hand = state.hand;
  seat($("bot-seat"), hand.players[1], false, hand);
  seat($("human-seat"), hand.players[0], true, hand);
  cards($("board"), hand.board, 5 - hand.board.length);
  $("street").textContent = hand.street.toUpperCase();
  $("pot").textContent = `POT · ${bb(hand.pot)}`;
  $("turn").textContent = state.phase === "complete" ? "Benchmark complete" : state.phase === "aborted" ? "Benchmark ended early" : state.phase === "finished" ? "Hand complete" : hand.actor === 0 ? "Your turn" : "Bot is thinking…";
  $("result").textContent = hand.result ? `This hand: ${signed(hand.result.humanChips)}${activeBenchmark ? "" : ` · Session: ${signed(state.sessionChips)}`}` : "";
  $("new-hand").hidden = state.phase !== "finished"; $("new-hand").disabled = busy || !!saved.pending;
  $("new-hand").textContent = activeBenchmark ? `Deal hand ${state.benchmark.completedHands + 1} / ${state.benchmark.targetHands}` : "Deal next hand";
  renderControls(hand);
  const events = $("events"); clear(events);
  for (const event of hand.events) events.append(node("div", "event", eventText(event)));
  events.scrollTop = events.scrollHeight;
  if (state.phase === "finished" && !isBenchmark) loadDiagnostics();
}
async function maybeAdvanceBot() {
  if (processingBot || busy || !state?.hand || ["finished", "complete", "aborted"].includes(state.phase) || state.hand.actor !== 1) return;
  processingBot = true;
  try {
    await mutate(`/api/sessions/${state.sessionId}/advance`, { handId: state.hand.id, revision: state.revision });
  } catch (_) { /* Retry remains available on reconnect. */ }
  finally { processingBot = false; }
}
$("gate-form").addEventListener("submit", async (event) => {
  event.preventDefault(); saved.token = $("token").value.trim(); persist();
  try { await recover(); } catch (_) { notice("Access denied", true); }
});
$("create").addEventListener("click", async () => {
  try {
    const playMode = document.querySelector('input[name="playMode"]:checked').value;
    const sessionType = document.querySelector('input[name="sessionType"]:checked').value;
    let body;
    if (sessionType === "benchmark") {
      const selected = $("target-hands").value;
      const raw = selected === "custom" ? $("custom-hands").value.trim() : selected;
      const targetHands = /^\d+$/.test(raw) ? Number(raw) : NaN;
      if (!Number.isInteger(targetHands) || targetHands < 1 || targetHands > 5000) {
        notice("Choose a whole-number benchmark target between 1 and 5000 hands.", true);
        return;
      }
      body = { sessionType, playMode, targetHands };
    } else {
      body = { sessionType, playMode, visibility: $("visibility").value };
    }
    if (availableModels.length) body.modelVersion = $("model-version").value;
    await mutate("/api/sessions", body);
  } catch (_) { /* Recoverable with the same key. */ }
});
$("model-version").addEventListener("change", () => {
  const model = availableModels.find(item => item.version === $("model-version").value);
  if (model) {
    $("setup-model").textContent = `${model.name} · HU20 · SHA-256 ${model.sha256}`;
    $("setup-title").textContent = `Play ${model.version}`;
  }
});
let casualVisibility = $("visibility").value;
function updateSetup() {
  const benchmark = document.querySelector('input[name="sessionType"]:checked').value === "benchmark";
  $("benchmark-options").hidden = !benchmark;
  $("visibility").value = benchmark ? "benchmark" : casualVisibility;
  $("visibility").disabled = benchmark;
  $("custom-hands").hidden = !benchmark || $("target-hands").value !== "custom";
  $("custom-hands-label").hidden = $("custom-hands").hidden;
}
for (const control of document.querySelectorAll('input[name="sessionType"]')) control.addEventListener("change", updateSetup);
$("visibility").addEventListener("change", () => { casualVisibility = $("visibility").value; });
$("target-hands").addEventListener("change", updateSetup);
updateSetup();
$("new-hand").addEventListener("click", async () => {
  if (!state) return;
  try {
    await mutate(`/api/sessions/${state.sessionId}/hands`, { revision: state.revision });
    await maybeAdvanceBot();
  } catch (_) { /* Recoverable with the same key. */ }
});
$("past-hands").addEventListener("click", async () => {
  if (!state) return;
  try {
    const history = await request(`/api/sessions/${state.sessionId}/history`);
    const events = $("events"); clear(events);
    if (!history.hands.length) events.append(node("div", "event", "No completed hands yet"));
    for (const hand of [...history.hands].reverse()) {
      const details = node("details", "past-hand");
      details.append(node("summary", "", `${hand.handId.slice(0, 8)}… · ${hand.humanChips === undefined ? "Result hidden during benchmark" : signed(hand.humanChips)} · ${hand.button === 0 ? "Button" : "Big blind"}`));
      for (const event of hand.events) details.append(node("div", "event", eventText(event)));
      events.append(details);
    }
    notice(`${history.hands.length} completed hands${state.sessionChips === undefined ? "" : ` · ${signed(state.sessionChips)} total`}`);
  } catch (error) { notice(error.message, true); }
});
$("benchmark-end").addEventListener("click", async () => {
  if (!state?.benchmark || state.benchmark.status !== "ACTIVE") return;
  if (!window.confirm("End this benchmark early? Completed hands will remain recorded, and the result will be marked INCOMPLETE.")) return;
  try {
    await mutate(`/api/sessions/${state.sessionId}/benchmark/end`, {
      revision: state.revision, handId: state.hand?.id || null, confirm: true
    });
  } catch (_) { /* A lost response can be retried with the same key. */ }
});
$("export-benchmark").addEventListener("click", async () => {
  if (!state?.benchmarkResult) return;
  try {
    const report = await request(`/api/sessions/${state.sessionId}/benchmark/export`);
    const blob = new Blob([JSON.stringify(report, null, 2) + "\n"], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = node("a"); link.href = url; link.download = `hu20-benchmark-${report.benchmarkId}.json`;
    document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (error) { notice(error.message, true); }
});
window.addEventListener("online", recover);
$("retry").addEventListener("click", recover);
recover();
