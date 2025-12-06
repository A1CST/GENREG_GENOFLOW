import argparse
import sys
import time

from main import ola_engine
from tinyllama_client import TinyLlamaClient


class OLATinyBridge:
    def __init__(self):
        # Two TinyLlama clients with independent histories
        self.convo = TinyLlamaClient(on_response=self._on_convo_response)
        self.voice = TinyLlamaClient(on_response=self._on_voice_response)
        self._last_auto = 0.0
        self._last_sent = 0.0
        self._last_recv = time.time()
        self._neutral_ready = True
        self._last_avg_trust = None
        self._send_count = 0
        self.last_sent_state = None
        # Set distinct system prompts if provided in config
        try:
            self.convo.system_prompt = self.convo.cfg.get(
                'convo_prompt',
                'You are the dialogue driver, keep discussion flowing naturally.'
            )
            self.voice.system_prompt = self.voice.cfg.get(
                'voice_prompt',
                'You are the voice of the OLA system. Express its thoughts clearly.'
            )
        except Exception:
            pass

    def _on_convo_response(self, text: str):
        if not text:
            return
        try:
            print("[Bridge] Convo -> OLA:", (text[:160] + '...') if len(text) > 160 else text)
        except Exception:
            pass
        try:
            ola_engine.feed_text(text, source='tinyllama')
        except Exception:
            pass
        # Track inbound activity
        self._last_recv = time.time()
        self._neutral_ready = True
        # Build concise OLA state for voice client
        try:
            mm = ola_engine.get_minimal_metrics()
            avg = float(mm.get('avg_trust', 0.0))
            tsd = float(mm.get('trust_std', 0.0))
            healthy = bool(mm.get('healthy', False))
            state_text = f"State update: avg_trust={avg:.3f}, trust_std={tsd:.3f}, healthy={'YES' if healthy else 'NO'}.".strip()
        except Exception:
            state_text = "State update available."
        outbound = state_text + "\n#ctx"
        # avoid repeating same status
        if self.last_sent_state != outbound:
            try:
                print("[Bridge] OLA -> Voice in:", (outbound[:160] + '...') if len(outbound) > 160 else outbound)
            except Exception:
                pass
            try:
                self.voice.submit_prompt(outbound)
                self.last_sent_state = outbound
            except Exception:
                pass

    def _on_voice_response(self, text: str):
        if not text:
            return
        # Sanitizer and final #ctx
        out = text.strip()
        try:
            out = out.replace("[TK]", "").strip()
        except Exception:
            pass
        if (not out) or (len(out) < 2):
            out = "Let's talk about something else."
        out += "\n#ctx"
        try:
            print("[Bridge] Voice -> User:", (out[:160] + '...') if len(out) > 160 else out)
        except Exception:
            pass
        try:
            ola_engine.feed_text(out, source='tinyllama')
        except Exception:
            pass
        self._last_sent = time.time()
        self._neutral_ready = False

    def start(self, use_stdin: bool, interval: float, warmup: bool, auto: bool):
        # Start TinyLlama workers
        self.convo.start()
        self.voice.start()

        # Optional warmup ping to verify server
        if warmup:
            cw = self.convo.chat_once(
                self.convo.cfg.get('convo_warmup', 'Hello from user. Please greet and ask one short question.'),
                update_history=True,
            )
            if cw:
                print("[Bridge] Convo Warmup OK:", (cw[:120] + '...') if len(cw) > 120 else cw)
            vw = self.voice.chat_once(
                self.voice.cfg.get('voice_warmup', 'You will voice OLA internal state succinctly.'),
                update_history=True,
            )
            if vw:
                print("[Bridge] Voice Warmup OK:", (vw[:120] + '...') if len(vw) > 120 else vw)

        if not use_stdin:
            # Idle loop — worker handles callbacks; optional neutral auto-prompt (inactivity-based)
            print("OLA↔TinyLlama bridge running (idle). Use --stdin to chat. Ctrl+C to stop.")
            try:
                while True:
                    now = time.time()
                    if auto and (now - self._last_auto) >= max(0.05, float(interval)):
                        self._last_auto = now
                        # Only send a neutral prompt if both sides have been inactive long enough and we haven't just sent one
                        inactivity = min(now - self._last_sent, now - self._last_recv)
                        threshold = max(3.0, float(interval) * 6.0)
                        if self._neutral_ready and inactivity >= threshold:
                            try:
                                neutral = "Let's continue talking about this.\n#ctx\n"
                                print("[Bridge] Auto -> TinyLlama:", neutral)
                                self.convo.submit_prompt(neutral)
                                self._last_sent = time.time()
                                self._neutral_ready = False
                            except Exception:
                                pass
                    time.sleep(max(0.05, float(interval)))
            except KeyboardInterrupt:
                return

        # Simple stdin chat loop
        print("[Bridge] Stdin mode active. Type messages and press Enter. Ctrl+C to stop.")
        try:
            while True:
                try:
                    line = input().strip()
                except EOFError:
                    break
                except KeyboardInterrupt:
                    break
                if not line:
                    continue
                try:
                    print("[Bridge] OLA -> TinyLlama:", (line[:160] + '...') if len(line) > 160 else line)
                except Exception:
                    pass
                try:
                    ola_engine.feed_text(line, source='user')
                except Exception:
                    pass
                # Forward to conversational TinyLlama; reply handled in _on_convo_response
                self.convo.submit_prompt(line)
                # Mark outbound activity and suppress immediate neutral prompt
                self._last_sent = time.time()
                self._neutral_ready = False
                # No auto prompting in stdin mode
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple OLA↔TinyLlama bridge")
    parser.add_argument("--stdin", action="store_true", help="Read user input from stdin and chat")
    parser.add_argument("--interval", type=float, default=1.0, help="Main loop sleep interval when idle")
    parser.add_argument("--no-warmup", action="store_true", help="Skip initial TinyLlama warmup request")
    parser.add_argument("--auto", action="store_true", help="Periodically send OLA compose_prompt() to TinyLlama")
    args = parser.parse_args()

    bridge = OLATinyBridge()
    bridge.start(use_stdin=args.stdin, interval=args.interval, warmup=(not args.no_warmup), auto=args.auto)


