// Load through the collaborative browser's evaluate tool after opening a session.
// Uses the visible controls and checks the actual DOM against retained decisions.
window.spectatorSmoke = {
  checks: [],
  async idle() {
    for (let i = 0; i < 500; i++) {
      if (!busy && !spectatorPlayback.busy) return;
      await new Promise(resolve => setTimeout(resolve, 10));
    }
    throw new Error("Browser action did not settle");
  },
  check() {
    const record = state.hand.decisions.at(-1);
    const rows = [...document.querySelectorAll('#decision-inspector .probability-table tbody tr')];
    if (rows.length !== record.menu.length || rows.some((row, i) =>
      row.cells[2].textContent !== String(record.menu[i].probability) ||
      (row.cells[3].textContent === '✓ Played') !== (i === record.selectedIndex))) throw new Error('Displayed distribution mismatch');
    if (!document.querySelector('#decision-inspector h3').textContent.includes(`Bot ${record.seat ? 'B' : 'A'}`)) throw new Error('Perspective label');
    if (record.observation.seat !== record.seat || record.model.sha256 !== state.models[record.seat].sha256) throw new Error('Bot identity');
    if (saved.pending || document.querySelector('#notice').classList.contains('error')) throw new Error('Browser error');
    this.checks.push({sessionId: state.sessionId, hand: state.hand.number, decision: record.number,
                      seat: record.seat, street: record.observation.street, lookup: record.lookup});
  },
  async step() {
    const before = state.phase === 'playing' ? state.hand.decisions.length : 0;
    document.querySelector('#spectator-step').click(); await this.idle();
    if (state.hand.decisions.length !== before + 1 || spectatorPlayback.running) throw new Error('Step semantics');
    this.check();
  },
  async pauseCheck() {
    document.querySelector('#spectator-play').click();
    await new Promise(resolve => setTimeout(resolve, 100));
    document.querySelector('#spectator-pause').click(); await this.idle();
    const revision = state.revision;
    await new Promise(resolve => setTimeout(resolve, 1800));
    if (state.revision !== revision || spectatorPlayback.running) throw new Error('Pause scheduled another decision');
    if (state.hand.decisions.length) this.check();
    return {status: 'passed', revision};
  },
};
