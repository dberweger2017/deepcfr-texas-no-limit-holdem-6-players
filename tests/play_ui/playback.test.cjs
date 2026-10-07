const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../../apps/poker-web/playback.js'), 'utf8');
const context = vm.createContext({ setTimeout, clearTimeout });
vm.runInContext(source + '\nglobalThis.Playback = SpectatorPlayback;', context);
const Playback = context.Playback;

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
function timers() {
  let next = 0;
  const waiting = new Map();
  return {
    setTimer: (run) => { waiting.set(++next, run); return next; },
    clearTimer: (id) => waiting.delete(id),
    size: () => waiting.size,
    fire: () => { const [id, run] = waiting.entries().next().value; waiting.delete(id); run(); },
  };
}
const flush = () => new Promise(setImmediate);

test('play queues one request, pause cancels the next, and rapid steps cannot overlap', async () => {
  const timer = timers(); const pending = deferred(); let calls = 0;
  const playback = new Playback({ advance: () => { calls++; return pending.promise; }, changed: () => {}, ...timer });
  playback.play(); playback.play(); assert.equal(timer.size(), 1);
  timer.fire(); assert.equal(calls, 1); assert.equal(playback.busy, true);
  await playback.step(); playback.play(); assert.equal(calls, 1);
  playback.pause(); pending.resolve(); await flush();
  assert.equal(timer.size(), 0); assert.equal(playback.running, false);
  const p = playback.step(); await playback.step(); await p;
  assert.equal(calls, 2); assert.equal(timer.size(), 0);
});

test('pause during an in-flight deal invalidates its automatic decision continuation', async () => {
  const timer = timers(); const dealt = deferred(); let decisions = 0;
  const playback = new Playback({ advance: async (allowed) => { await dealt.promise; if (allowed()) decisions++; }, changed: () => {}, ...timer });
  playback.play(); timer.fire(); playback.pause(); dealt.resolve(); await flush();
  assert.equal(decisions, 0); assert.equal(timer.size(), 0);
  await playback.step(); assert.equal(decisions, 1);
});

test('play schedules only after completion and errors leave playback paused', async () => {
  const timer = timers(); const pending = deferred();
  const playback = new Playback({ advance: () => pending.promise, changed: () => {}, ...timer });
  playback.play(); timer.fire(); assert.equal(timer.size(), 0);
  pending.resolve(); await flush(); assert.equal(timer.size(), 1);
  playback.pause(); assert.equal(timer.size(), 0);
  const broken = new Playback({ advance: async () => { throw new Error('lost response'); }, changed: () => {}, ...timer });
  broken.play(); timer.fire(); await flush();
  assert.equal(broken.running, false); assert.equal(broken.busy, false); assert.equal(timer.size(), 0);
});
