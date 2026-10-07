"use strict";

// The next request is scheduled only after the current one settles. Pause
// invalidates queued work and any continuation after an in-flight deal.
class SpectatorPlayback {
  constructor({ advance, changed, delay = () => 1500, setTimer = setTimeout, clearTimer = clearTimeout }) {
    this.advance = advance; this.changed = changed; this.delay = delay;
    this.setTimer = setTimer; this.clearTimer = clearTimer;
    this.running = false; this.busy = false; this.timer = null;
  }
  play() {
    if (this.running || this.busy) return;
    this.running = true; this.changed(); this.schedule(0);
  }
  pause() {
    this.running = false;
    if (this.timer !== null) this.clearTimer(this.timer);
    this.timer = null; this.changed();
  }
  schedule(delay) {
    if (!this.running) return;
    this.timer = this.setTimer(() => { this.timer = null; this.run(true); }, delay);
  }
  async step() {
    if (this.busy) return;
    this.pause(); await this.run(false);
  }
  async run(automatic) {
    if (this.busy || (automatic && !this.running)) return;
    this.busy = true; this.changed();
    try { await this.advance(() => !automatic || this.running); }
    catch (_) { this.pause(); }
    finally {
      this.busy = false; this.changed();
      if (this.running) this.schedule(this.delay());
    }
  }
}
