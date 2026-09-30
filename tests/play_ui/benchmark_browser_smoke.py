"""Bounded M4 Chrome smoke for the benchmark setup and finished result screen."""

import argparse
import base64
import json
import sqlite3
import subprocess
import time
from pathlib import Path
from urllib.request import urlopen

from src.play_api.service import MODEL_SHA256
from tests.play_ui.browser_smoke import BASE, CHROME, PORT, CDP


def screenshot(cdp, path):
    image = cdp.call("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": True})
    path.write_bytes(base64.b64decode(image["data"]))


def run_mode(cdp, token, mode, database, output):
    cdp.call("Page.navigate", {"url": BASE})
    cdp.wait("document.readyState === 'complete' && !!document.querySelector('#gate-form')")
    cdp.js(f"document.querySelector('#token').value = {json.dumps(token)}; document.querySelector('#gate-form').requestSubmit()")
    cdp.wait("!document.querySelector('#setup').hidden")
    cdp.wait(f"document.querySelector('#setup-model').textContent.includes('{MODEL_SHA256}')")
    cdp.js("document.querySelector('input[name=sessionType][value=benchmark]').click()")
    cdp.js("document.querySelector('#target-hands').value = 'custom'; document.querySelector('#target-hands').dispatchEvent(new Event('change'))")
    cdp.js("document.querySelector('#custom-hands').value = '1'")
    if mode == "free":
        cdp.js("document.querySelector('input[name=playMode][value=free]').click()")
    assert cdp.js("!document.querySelector('#benchmark-options').hidden")
    if mode == "restricted":
        screenshot(cdp, output / "benchmark-setup.png")
    cdp.js("document.querySelector('#create').click()")
    cdp.wait("!document.querySelector('#game').hidden && !document.querySelector('#new-hand').hidden")
    assert cdp.js("document.querySelector('#session-stat').hidden")
    cdp.js("document.querySelector('#new-hand').click()")
    cdp.wait("document.querySelector('#turn').textContent === 'Your turn'")
    assert cdp.js("document.querySelector('#progress-text').textContent.includes('Hand 1 / 1')")
    if mode == "free":
        cdp.js("document.querySelector('.raise-row input[type=text]').value = '2.01'; document.querySelector('.raise-row .primary').click()")
    for _ in range(100):
        if cdp.js("document.querySelector('#turn').textContent === 'Benchmark complete'"):
            break
        if cdp.js("document.querySelector('#turn').textContent === 'Your turn' && !document.querySelector('#controls button:disabled')"):
            cdp.js("(() => { const b=[...document.querySelectorAll('#controls button')].find(x=>/^(check|call)/i.test(x.textContent)); if(b) b.click(); })()")
        time.sleep(0.15)
    else:
        raise TimeoutError(f"{mode} benchmark browser hand did not finish")
    assert cdp.js("!document.querySelector('#benchmark-results').hidden")
    assert cdp.js("document.querySelector('#benchmark-status').textContent === 'COMPLETE'")
    assert cdp.js("document.querySelector('#benchmark-summary').textContent.includes('Raw BB/100')")
    screenshot(cdp, output / f"benchmark-{mode}-result.png")
    session = cdp.js("JSON.parse(localStorage.getItem('hu20-local-play-v1')).sessionId")
    private = json.loads(sqlite3.connect(database).execute(
        "SELECT state FROM sessions WHERE id=?", (session,)).fetchone()[0])
    assert private["handsPlayed"] == private["benchmark"]["targetHands"] == 1
    if mode == "free":
        assert private["history"][0]["actions"][0]["raiseTo"] == 201
    cdp.js("localStorage.setItem('hu20-local-play-v1', JSON.stringify({token: JSON.parse(localStorage.getItem('hu20-local-play-v1')).token}))")
    return session


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("results/play-web"))
    args = parser.parse_args()
    output = args.data_dir / "screenshots"
    output.mkdir(parents=True, exist_ok=True)
    token = (args.data_dir / "access.token").read_text().strip()
    process = subprocess.Popen([CHROME, "--headless=new", "--no-first-run", "--disable-gpu",
        "--window-size=1280,900",
        f"--user-data-dir={(args.data_dir / 'chrome-benchmark-profile').resolve()}",
        f"--remote-debugging-port={PORT}", "--remote-allow-origins=*", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                pages = json.load(urlopen(f"http://127.0.0.1:{PORT}/json", timeout=1))
                page = next(item for item in pages if item["type"] == "page")
                break
            except Exception:
                time.sleep(0.1)
        else:
            raise RuntimeError("Chrome debugging endpoint did not start")
        cdp = CDP(page["webSocketDebuggerUrl"])
        try:
            cdp.call("Page.enable")
            cdp.call("Runtime.enable")
            for mode in ("restricted", "free"):
                session = run_mode(cdp, token, mode, args.data_dir / "private.sqlite", output)
                print(json.dumps({"mode": mode, "session": session,
                                  "screenshot": str(output / f"benchmark-{mode}-result.png")}))
        finally:
            cdp.socket.close()
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    main()
