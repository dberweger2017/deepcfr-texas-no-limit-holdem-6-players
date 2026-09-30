"""Bounded M4 Chrome smoke; run explicitly, never in the regular test suite."""

import base64
import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, str(Path("results/browser-deps").resolve()))
import websocket  # type: ignore[import-not-found]

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
BASE = "http://127.0.0.1:8767"
PORT = 9227


class CDP:
    def __init__(self, url):
        self.socket = websocket.create_connection(url, timeout=15, origin=f"http://127.0.0.1:{PORT}")
        self.next_id = 0

    def call(self, method, params=None):
        self.next_id += 1
        expected = self.next_id
        self.socket.send(json.dumps({"id": expected, "method": method, "params": params or {}}))
        while True:
            message = json.loads(self.socket.recv())
            if message.get("id") == expected:
                if "error" in message:
                    raise RuntimeError(message["error"])
                return message.get("result", {})

    def js(self, expression):
        response = self.call("Runtime.evaluate", {"expression": expression,
                     "returnByValue": True, "awaitPromise": True})
        if "exceptionDetails" in response:
            raise RuntimeError(response["exceptionDetails"])
        return response["result"].get("value")

    def wait(self, expression, timeout=15):
        until = time.time() + timeout
        while time.time() < until:
            if self.js(expression):
                return
            time.sleep(0.15)
        raise TimeoutError(expression)


def run_mode(cdp, token, mode, output):
    cdp.call("Page.navigate", {"url": BASE})
    cdp.wait("document.readyState === 'complete' && !!document.querySelector('#gate-form')")
    # Token is supplied through the form, never a URL or console log.
    cdp.js(f"document.querySelector('#token').value = {json.dumps(token)}; document.querySelector('#gate-form').requestSubmit()")
    cdp.wait("!document.querySelector('#setup').hidden")
    if mode == "free":
        cdp.js("document.querySelector('input[name=playMode][value=free]').click()")
    cdp.js("document.querySelector('#create').click()")
    cdp.wait("!document.querySelector('#game').hidden && !document.querySelector('#new-hand').hidden")
    cdp.js("document.querySelector('#new-hand').click()")
    cdp.wait("document.querySelector('#turn').textContent === 'Your turn'")
    if mode == "free":
        cdp.wait("!!document.querySelector('.raise-row input[type=text]')")
    screenshot = cdp.call("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": True})
    (output / f"{mode}-table.png").write_bytes(base64.b64decode(screenshot["data"]))
    if mode == "free":
        cdp.js("document.querySelector('.raise-row input[type=text]').value = '2.01'; document.querySelector('.raise-row .primary').click()")
    for _ in range(100):
        if cdp.js("document.querySelector('#turn').textContent === 'Hand complete'"):
            break
        if cdp.js("document.querySelector('#turn').textContent === 'Your turn' && [...document.querySelectorAll('#controls button')].some(x=>!x.disabled && /^(check|call)/i.test(x.textContent))"):
            cdp.js("(() => { const b=[...document.querySelectorAll('#controls button')].find(x=>!x.disabled && /^(check|call)/i.test(x.textContent)); if(b) b.click(); })()")
        time.sleep(0.12)
    else:
        raise TimeoutError(f"{mode} browser hand did not finish")
    assert cdp.js("document.querySelector('#result').textContent.includes('This hand:')")
    session = cdp.js("JSON.parse(localStorage.getItem('hu20-local-play-v1')).sessionId")
    # Clear only this smoke browser's session; keep the access token for the second mode.
    cdp.js("localStorage.setItem('hu20-local-play-v1', JSON.stringify({token: JSON.parse(localStorage.getItem('hu20-local-play-v1')).token}))")
    return session


def main():
    output = Path("results/play-web/screenshots")
    output.mkdir(parents=True, exist_ok=True)
    token = Path("results/play-web/access.token").read_text().strip()
    process = subprocess.Popen([CHROME, "--headless=new", "--no-first-run", "--disable-gpu", "--window-size=1280,900",
        f"--user-data-dir={Path('results/play-web/chrome-profile').resolve()}",
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
                session = run_mode(cdp, token, mode, output)
                print(json.dumps({"mode": mode, "session": session, "screenshot": str(output / f"{mode}-table.png")}))
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
