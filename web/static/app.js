const SEAT_POSITIONS = {
  0: { left: "50%", top: "88%" },
  1: { left: "14%", top: "68%" },
  2: { left: "10%", top: "34%" },
  3: { left: "50%", top: "12%" },
  4: { left: "90%", top: "34%" },
  5: { left: "86%", top: "68%" },
};

let sessionId = null;

const els = {
  modelsDir: document.getElementById("models-dir"),
  heroSeat: document.getElementById("hero-seat"),
  startSession: document.getElementById("start-session"),
  setupPanel: document.getElementById("setup-panel"),
  controlsPanel: document.getElementById("controls-panel"),
  seats: document.getElementById("seats"),
  communityCards: document.getElementById("community-cards"),
  potAmount: document.getElementById("pot-amount"),
  sessionProfit: document.getElementById("session-profit"),
  handNumber: document.getElementById("hand-number"),
  stageLabel: document.getElementById("stage-label"),
  turnIndicator: document.getElementById("turn-indicator"),
  actionLog: document.getElementById("action-log"),
  opponentsList: document.getElementById("opponents-list"),
  loadWarnings: document.getElementById("load-warnings"),
  handResult: document.getElementById("hand-result"),
  newHand: document.getElementById("new-hand"),
  toggleReveal: document.getElementById("toggle-reveal"),
  raiseAmount: document.getElementById("raise-amount"),
  raiseSubmit: document.getElementById("raise-submit"),
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || "Request failed");
  }
  return response.json();
}

function money(value) {
  const sign = value >= 0 ? "" : "-";
  return `${sign}$${Math.abs(value).toFixed(2)}`;
}

function renderCard(card, hidden = false) {
  const div = document.createElement("div");
  if (hidden || !card) {
    div.className = "card back";
    div.textContent = "🂠";
    return div;
  }
  div.className = `card ${card.color}`;
  div.innerHTML = `<span class="rank">${card.rank}</span><span class="suit">${card.suit}</span>`;
  return div;
}

function renderSeats(state) {
  els.seats.innerHTML = "";
  if (!state.players) return;

  for (const player of state.players) {
    const seat = document.createElement("div");
    seat.className = "seat";
    if (player.is_current) seat.classList.add("current");
    if (player.is_hero) seat.classList.add("hero");
    if (!player.active) seat.classList.add("inactive");

    const pos = SEAT_POSITIONS[player.seat] || SEAT_POSITIONS[0];
    seat.style.left = pos.left;
    seat.style.top = pos.top;

    seat.innerHTML = `
      <div class="seat-header">
        <span class="seat-name">${player.name}${player.is_button ? " • BTN" : ""}</span>
        <span class="seat-stack">${money(player.stack)}</span>
      </div>
      <div class="seat-meta">
        <span>Bet ${money(player.bet)}</span>
        <span>${player.active ? "In hand" : "Folded"}</span>
      </div>
      <div class="seat-cards"></div>
    `;

    const cardsWrap = seat.querySelector(".seat-cards");
    if (player.cards && player.cards.length) {
      player.cards.forEach((card) => cardsWrap.appendChild(renderCard(card)));
    } else {
      cardsWrap.appendChild(renderCard(null, true));
      cardsWrap.appendChild(renderCard(null, true));
    }

    els.seats.appendChild(seat);
  }
}

function renderCommunity(state) {
  els.communityCards.innerHTML = "";
  const cards = state.community_cards || [];
  if (!cards.length) {
    const placeholder = document.createElement("div");
    placeholder.className = "turn-indicator";
    placeholder.textContent = "Waiting for board";
    els.communityCards.appendChild(placeholder);
    return;
  }
  cards.forEach((card) => els.communityCards.appendChild(renderCard(card)));
}

function renderActionLog(state) {
  els.actionLog.innerHTML = "";
  (state.action_log || []).slice().reverse().forEach((entry) => {
    const li = document.createElement("li");
    li.textContent = entry.label || `${entry.action} (${entry.seat})`;
    els.actionLog.appendChild(li);
  });
}

function renderOpponents(state) {
  els.opponentsList.innerHTML = "";
  (state.opponents || []).forEach((opp) => {
    const li = document.createElement("li");
    li.textContent = `Seat ${opp.seat}: ${opp.name}`;
    els.opponentsList.appendChild(li);
  });

  const warnings = state.load_warnings || [];
  if (!warnings.length) {
    els.loadWarnings.classList.add("hidden");
    els.loadWarnings.innerHTML = "";
    return;
  }
  els.loadWarnings.classList.remove("hidden");
  els.loadWarnings.innerHTML = "";
  warnings.forEach((warning) => {
    const li = document.createElement("li");
    li.textContent = `Seat ${warning.seat}: failed to load ${warning.checkpoint} (${warning.error})`;
    els.loadWarnings.appendChild(li);
  });
}

function setActionButtons(state) {
  const enabled = state.is_hero_turn;
  document.querySelectorAll(".action-btn").forEach((button) => {
    const action = button.dataset.action;
    const allowed = (state.legal_actions || []).includes(action);
    button.disabled = !enabled || !allowed;
    if (action === "call" && state.raise_bounds?.call_amount != null) {
      button.textContent = `Call ${money(state.raise_bounds.call_amount)}`;
    } else if (action === "call") {
      button.textContent = "Call";
    }
  });
  els.raiseSubmit.disabled = !enabled || !(state.legal_actions || []).includes("raise");
  els.newHand.disabled = !sessionId;
  els.toggleReveal.disabled = !sessionId || !state.has_active_hand;

  if (state.raise_bounds) {
    els.raiseAmount.min = state.raise_bounds.min_raise;
    els.raiseAmount.max = state.raise_bounds.max_raise;
    els.raiseAmount.placeholder = `Raise ${state.raise_bounds.min_raise.toFixed(0)} - ${state.raise_bounds.max_raise.toFixed(0)}`;
  }
}

function renderState(state) {
  els.sessionProfit.textContent = money(state.session_profit || 0);
  els.handNumber.textContent = state.hand_number || "—";
  els.stageLabel.textContent = state.stage || "—";
  els.potAmount.textContent = money(state.pot || 0);

  renderSeats(state);
  renderCommunity(state);
  renderActionLog(state);
  renderOpponents(state);
  setActionButtons(state);

  if (state.is_hero_turn) {
    els.turnIndicator.textContent = "Your turn — choose an action";
    els.turnIndicator.classList.add("active");
  } else if (state.final_state) {
    els.turnIndicator.textContent = "Hand complete";
    els.turnIndicator.classList.remove("active");
  } else {
    els.turnIndicator.textContent = "AI thinking…";
    els.turnIndicator.classList.remove("active");
  }

  if (state.hand_result) {
    els.handResult.classList.remove("hidden");
    els.handResult.classList.toggle("win", state.hand_result.won);
    els.handResult.classList.toggle("loss", !state.hand_result.won);
    els.handResult.textContent = state.hand_result.won
      ? `You won ${money(state.hand_result.hero_reward)}`
      : `You lost ${money(state.hand_result.hero_reward)}`;
  } else {
    els.handResult.classList.add("hidden");
  }
}

async function loadModelDirs() {
  const data = await api("/api/model-dirs");
  els.modelsDir.innerHTML = "";
  const dirs = data.model_dirs?.length ? data.model_dirs : ["models/standard/phase1"];
  dirs.forEach((dir) => {
    const option = document.createElement("option");
    option.value = dir;
    option.textContent = dir;
    els.modelsDir.appendChild(option);
  });
}

async function startSession() {
  els.startSession.disabled = true;
  els.startSession.textContent = "Loading models…";
  try {
    const payload = {
      models_dir: els.modelsDir.value,
      hero_seat: Number(els.heroSeat.value),
    };
    const state = await api("/api/sessions", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    sessionId = state.session_id;
    els.setupPanel.classList.add("hidden");
    els.controlsPanel.classList.remove("hidden");
    renderState(state);
    await newHand();
  } finally {
    els.startSession.disabled = false;
    els.startSession.textContent = "Join table";
  }
}

async function newHand() {
  if (!sessionId) return;
  const state = await api(`/api/sessions/${sessionId}/new-hand`, { method: "POST" });
  renderState(state);
}

async function sendAction(action, amount = null) {
  if (!sessionId) return;
  const payload = { action };
  if (amount !== null) payload.amount = amount;
  const state = await api(`/api/sessions/${sessionId}/action`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
  renderState(state);
}

els.startSession.addEventListener("click", () => {
  startSession().catch((error) => {
    alert(error.message);
  });
});

els.newHand.addEventListener("click", () => {
  newHand().catch((error) => alert(error.message));
});

els.toggleReveal.addEventListener("click", () => {
  api(`/api/sessions/${sessionId}/reveal`, { method: "POST" })
    .then(renderState)
    .catch((error) => alert(error.message));
});

document.querySelectorAll(".action-btn").forEach((button) => {
  button.addEventListener("click", () => {
    sendAction(button.dataset.action).catch((error) => alert(error.message));
  });
});

els.raiseSubmit.addEventListener("click", () => {
  const amount = Number(els.raiseAmount.value);
  if (!amount) return;
  sendAction("raise", amount).catch((error) => alert(error.message));
});

loadModelDirs().catch((error) => {
  console.error(error);
});
