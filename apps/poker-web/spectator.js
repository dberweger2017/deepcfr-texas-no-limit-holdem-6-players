"use strict";

const spectatorPlayback = new SpectatorPlayback({
  advance: async (continueAllowed) => {
    if (!state || state.sessionType !== "spectator" || busy || saved.pending) throw new Error("Spectator unavailable");
    if (!state.hand || state.phase === "finished") {
      await mutate(`/api/sessions/${state.sessionId}/hands`, { revision: state.revision });
      if (!continueAllowed()) return;
    }
    await mutate(`/api/sessions/${state.sessionId}/advance`, { handId: state.hand.id, revision: state.revision });
  },
  changed: () => { if (state?.sessionType === "spectator") render(); },
  delay: () => Number($("spectator-speed").value),
});

function setupSpectator(catalog) {
  const input = document.querySelector('input[name="sessionType"][value="spectator"]');
  input.disabled = !catalog.spectatorAvailable;
  $("spectator-choice").hidden = !catalog.spectatorAvailable;
  for (const id of ["bot-a-model", "bot-b-model"]) {
    clear($(id));
    for (const item of catalog.models) {
      const option = node("option", "", item.name); option.value = item.version; $(id).append(option);
    }
  }
  $("bot-a-model").value = catalog.default || "";
  $("bot-b-model").value = catalog.models.find(item => item.version !== catalog.default)?.version || catalog.default || "";
  for (const id of ["bot-a-model", "bot-b-model"]) $(id).onchange = spectatorIdentities;
  spectatorIdentities();
}

function spectatorIdentities() {
  clear($("spectator-identities"));
  for (const [index, id] of ["bot-a-model", "bot-b-model"].entries()) {
    const model = availableModels.find(item => item.version === $(id).value);
    if (model) $("spectator-identities").append(node("div", "identity", `Bot ${index ? "B" : "A"} · ${model.version}\nModel SHA-256 ${model.sha256}\nManifest SHA-256 ${model.manifestSha256}`));
  }
}

function spectatorSeat(target, seatIndex, hand) {
  const view = hand.perspectives[seatIndex];
  const player = view.players[seatIndex];
  clear(target); target.classList.toggle("acting", hand.actor === seatIndex);
  const head = node("div", "seat-head");
  head.append(node("strong", "", `Bot ${seatIndex ? "B" : "A"} · ${state.models[seatIndex].version}`));
  head.append(node("span", "badge", hand.button === seatIndex ? "D · SB" : "BB"));
  target.append(head, node("div", "stack", bb(player.stack)));
  const deck = node("div", "cards seat-cards"); cards(deck, view.hole_cards); target.append(deck);
  target.append(node("small", "seat-detail", `${view.player_id} perspective · own cards${player.folded ? " · Folded" : player.stack === 0 ? " · All-in" : ""}`));
}

function identityDetails(models) {
  const fragment = document.createDocumentFragment();
  for (const [index, model] of models.entries()) {
    const item = node("div", "identity", `Bot ${index ? "B" : "A"} · ${model.name}\nModel SHA-256 ${model.sha256}\nManifest SHA-256 ${model.manifestSha256}\n${model.game} · ${model.schema} · ${model.adapter}\n`);
    const link = node("a", "", "Published release manifest"); link.href = model.manifestUrl;
    link.target = "_blank"; link.rel = "noopener noreferrer"; item.append(link); fragment.append(item);
  }
  return fragment;
}

function decisionPanel(record) {
  const view = record.observation;
  const panel = node("div", "decision-panel");
  panel.append(node("h3", "", `Decision ${record.number + 1} · Bot ${record.seat ? "B" : "A"} · ${record.model.version} · ${view.street}`));
  panel.append(node("div", "lookup", `Lookup: ${record.lookup}`));
  const ownCards = node("div", "cards seat-cards"); cards(ownCards, view.hole_cards); panel.append(ownCards);
  panel.append(node("p", "", `${view.player_id} perspective · Board: ${view.board.join(" ") || "none"} · Pot ${bb(view.pots.reduce((sum, pot) => sum + pot.amount, 0))}`));
  for (const player of view.players) panel.append(node("div", "", `Bot ${player.seat ? "B" : "A"}: stack ${bb(player.stack)} · street bet ${bb(player.street_bet)} · contributed ${bb(player.contributed)}${player.shown_cards.length ? ` · shown ${player.shown_cards.join(" ")}` : ""}`));
  const legal = view.legal_actions;
  panel.append(node("p", "", `Legal: ${legal.kinds.join(", ")} · call ${bb(legal.call_amount)}${legal.min_raise_to === null ? "" : ` · raise to ${bb(legal.min_raise_to)}–${bb(legal.max_raise_to)}`}`));
  const table = node("table", "probability-table");
  const heading = node("tr");
  for (const label of ["Action", "Raise to", "Probability", "Selected"]) heading.append(node("th", "", label));
  const thead = node("thead"); thead.append(heading); table.append(thead);
  const tbody = node("tbody");
  for (const [index, item] of record.menu.entries()) {
    const row = node("tr", index === record.selectedIndex ? "selected-action" : "");
    for (const text of [item.label, item.raiseTo === null ? "—" : `${bb(item.raiseTo)} (${item.raiseTo} chips)`, String(item.probability), index === record.selectedIndex ? "✓ Played" : ""]) row.append(node("td", "", text));
    tbody.append(row);
  }
  table.append(tbody); panel.append(table);
  const events = node("details"); events.append(node("summary", "", "Observed actions and amounts"));
  for (const event of view.history) {
    let text;
    if (event.event === "ActionTaken") text = `Bot ${event.seat ? "B" : "A"} · ${event.street} · ${event.action.kind}${event.action.raise_to === null ? "" : ` to ${bb(event.action.raise_to)}`} · paid ${bb(event.paid)}`;
    else if (event.event === "BlindPosted") text = `Bot ${event.seat ? "B" : "A"} posted ${bb(event.amount)}`;
    else if (event.event === "BoardDealt") text = `${event.street} · ${event.cards.join(" ")}`;
    else if (event.event === "CardsShown") text = `Bot ${event.seat ? "B" : "A"} showed ${event.cards.join(" ")}`;
    if (text) events.append(node("div", "event", text));
  }
  panel.append(events);
  const exact = node("details"); exact.append(node("summary", "", "Exact observation and decision JSON"));
  exact.append(node("pre", "observation-json", JSON.stringify(record, null, 2))); panel.append(exact);
  return panel;
}

function decisionHistory(records, target) {
  for (const record of records) {
    const details = node("details", "past-decision");
    const chosen = record.menu[record.selectedIndex];
    details.append(node("summary", "", `${record.number + 1} · Bot ${record.seat ? "B" : "A"} · ${record.observation.street} · ${chosen.label}${chosen.raiseTo === null ? "" : ` to ${bb(chosen.raiseTo)}`} · ${record.lookup}`));
    details.addEventListener("toggle", () => {
      if (details.open && details.childElementCount === 1) details.append(decisionPanel(record));
    });
    target.append(details);
  }
}

function renderSpectator() {
  const hand = state.hand;
  $("mode").textContent = `BOT-VS-BOT SPECTATOR / ${spectatorPlayback.running ? "PLAYING" : "PAUSED"} / 20 BB`;
  $("model").textContent = `Bot A · ${state.models[0].version} versus Bot B · ${state.models[1].version}`;
  $("session-stat").hidden = false; $("session-bb").textContent = `A ${signed(state.sessionChips[0])} · B ${signed(state.sessionChips[1])}`;
  $("progress").hidden = true; $("benchmark-end").hidden = true;
  renderBenchmarkResult(null);
  $("new-session").disabled = busy || !!saved.pending || spectatorPlayback.busy;
  $("new-hand").hidden = true;
  $("past-hands").disabled = busy || !!saved.pending || spectatorPlayback.busy;
  $("diagnostics").textContent = "";
  $("model-details").replaceChildren(node("div", "", `Session ${state.sessionId} · ${state.protocol} · Source ${state.sourceVersion}`), identityDetails(state.models));
  $("spectator-controls").hidden = false;
  $("spectator-play").disabled = spectatorPlayback.running || spectatorPlayback.busy || busy || !!saved.pending;
  $("spectator-pause").disabled = !spectatorPlayback.running;
  $("spectator-step").disabled = spectatorPlayback.running || spectatorPlayback.busy || busy || !!saved.pending;
  $("turn").textContent = hand ? (state.phase === "finished" ? `Hand ${hand.number + 1} complete` : `Bot ${hand.actor ? "B" : "A"} acts next`) : "Ready to watch";
  $("result").textContent = hand?.result ? `This hand: A ${signed(hand.result.netChips[0])} · B ${signed(hand.result.netChips[1])}` : "Each step plays one decision. Stacks reset and the button alternates each hand.";
  clear($("controls"));
  const events = $("events"); clear(events);
  const inspector = $("decision-inspector"); clear(inspector); $("spectator-inspector").hidden = false;
  if (!hand) {
    for (const id of ["bot-seat", "human-seat", "board"]) clear($(id));
    $("street").textContent = "PRE-FLOP"; $("pot").textContent = "POT · 0 BB";
    inspector.append(node("p", "", "Press Step or Play to inspect the first decision.")); return;
  }
  spectatorSeat($("bot-seat"), 1, hand); spectatorSeat($("human-seat"), 0, hand);
  cards($("board"), hand.board, 5 - hand.board.length);
  $("street").textContent = hand.street.toUpperCase(); $("pot").textContent = `POT · ${bb(hand.pot)}`;
  for (const event of hand.events) {
    const text = eventText(event).replace(/^You\b/, "Bot A").replace(/^Bot\b(?! A)/, "Bot B");
    events.append(node("div", "event", text));
  }
  events.scrollTop = events.scrollHeight;
  const latest = hand.decisions.at(-1);
  if (latest) inspector.append(decisionPanel(latest));
  else inspector.append(node("p", "", "The hand is dealt. Step plays the next bot decision."));
  const history = node("details", "past-hand"); history.append(node("summary", "", `This hand · ${hand.decisions.length} decisions`));
  decisionHistory(hand.decisions, history); inspector.append(history);
}
