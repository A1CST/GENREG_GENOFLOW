import os
import sys
import json
import time
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt
from rich import box
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Ensure local imports resolve within isolated_OLA only
CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from main import ola_engine  # noqa: E402


class OLAChatbot:
    def __init__(self, cfg_path: str = "ai_config.json", mode: int = 3):
        cfg_file = os.path.join(CURRENT_DIR, cfg_path) if not os.path.isabs(cfg_path) else cfg_path
        self.cfg = {}
        if os.path.exists(cfg_file):
            with open(cfg_file, "r", encoding="utf-8") as f:
                self.cfg = json.load(f)

        model_name = self.cfg.get("model", "openai-community/gpt2-medium")
        prefer = self.cfg.get("device", "cuda")
        use_cuda = (prefer == "cuda" and torch.cuda.is_available())
        self.device = "cuda" if use_cuda else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        print("[Chatbot] Loading model:", model_name, "on", self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        self.model.to(self.device)

        self.history = []
        self.mode = int(mode)

    def llm_reply(self, prompt: str, temperature: float, max_new: int) -> str:
        # Strip any embedded telemetry blocks from prompt before tokenization
        prompt_clean = self._strip_telemetry(prompt)
        ids = self.tokenizer(prompt_clean, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **ids,
            max_new_tokens=max_new,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.tokenizer.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)

    @staticmethod
    def _strip_telemetry(text: str) -> str:
        try:
            start_tag = "<<TELEMETRY>>"
            end_tag = "<<END_TELEMETRY>>"
            out = []
            i = 0
            n = len(text)
            while i < n:
                s = text.find(start_tag, i)
                if s == -1:
                    out.append(text[i:])
                    break
                out.append(text[i:s])
                e = text.find(end_tag, s + len(start_tag))
                if e == -1:
                    # no closing tag; drop rest
                    break
                i = e + len(end_tag)
            return "".join(out)
        except Exception:
            return text

    def chat(self, user_input: str) -> str:
        # Push user input into OLA sensory stream
        ola_engine.feed_text(user_input, source='user')

        # Get trust metric for parameter control only (do not inject into prompt)
        avg_trust = ola_engine.get_avg_trust()

        # Derive generation parameters from OLA trust (raise to 0.8–0.9)
        temperature = 0.8 + 0.1 * (1.0 - float(avg_trust))
        temperature = max(0.8, min(0.9, float(temperature)))
        max_new = int(self.cfg.get("max_new_tokens", 100))
        # Prompt strategy by mode
        if self.mode == 1:
            prefix = ""
        else:
            prefix = "The assistant is calm, helpful, and factual.\n"
        # Build short history tail (last 3 turns)
        try:
            tail_pairs = self.history[-3:]
            formatted = []
            for (u, r) in tail_pairs:
                formatted.append(f"User:{u}\nAssistant:{r}")
            history_tail = ("\n".join(formatted) + "\n") if formatted else ""
        except Exception:
            history_tail = ""
        # No system telemetry or env lines in GPT-2 prompt
        prompt = prefix + history_tail + f"User:{user_input}\nAssistant:"

        # Generate reply
        reply = self.llm_reply(prompt, temperature=temperature, max_new=max_new)
        # Safety moderation for modes 3 and 4
        if self.mode in (3, 4):
            low = reply.lower()
            if ("hit" in low) or ("kill" in low):
                try:
                    ola_engine.decrease_trust(0.2)
                    try:
                        ola_engine.log_vector("suppressed", ola_engine.get_action_vector())
                    except Exception:
                        pass
                except Exception:
                    pass
                reply = "[Response suppressed: incoherent pattern]"
        # Post-filter to remove echoed prompt tails
        try:
            reply = reply.split("User:")[0].strip()
        except Exception:
            pass
        # Remove invalid token artifacts
        try:
            reply = reply.replace("[TK]", "").strip()
        except Exception:
            pass
        if (not reply) or (len(reply.strip()) < 2):
            reply = "Let's talk about something else."
        self.history.append((user_input, reply))

        # Feed reply back into OLA for self-training
        ola_engine.feed_text(reply, source='assistant')
        return reply


def claude_style_console():
    console = Console()

    # Load ASCII logo
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.txt")
    if os.path.exists(logo_path):
        with open(logo_path, "r", encoding="utf-8") as f:
            logo_ascii = f.read()
    else:
        logo_ascii = "[ASCII logo missing]"

    title = "[bold red]OLA Chat v2.0.30[/bold red]"
    left = f"{logo_ascii}\n[bold white]Welcome back Payton![/bold white]\n\n[dim]OLA Engine • v1.0\n{os.getcwd()}[/dim]"

    right = (
        "[bold orange3]Tips for getting started[/bold orange3]\n"
        "[white]Run /init to load your OLA model configuration.[/white]\n"
        "[dim]Note: You have launched OLA in your home directory.[/dim]\n\n"
        "[bold orange3]Recent activity[/bold orange3]\n"
        "[dim]No recent activity[/dim]"
    )

    table = Table.grid(expand=True)
    table.add_column(justify="left", ratio=1)
    table.add_column(justify="left", ratio=2)
    table.add_row(left, right)

    console.print(Panel(table, title=title, border_style="red", box=box.SQUARE))
    console.print("[dim]> Try '/stats' to view engine stats[/dim]", justify="left")


def claude_chat_ui(bot: OLAChatbot):
    console = Console()
    console.print("")  # spacing
    while True:
        # user input panel
        user_msg = Prompt.ask("[white]>[/white]")
        if user_msg.lower() in {"exit", "quit"}:
            break
        if user_msg.lower() in {"/stats", "/s"}:
            try:
                stats = ola_engine.get_stats_text()
            except Exception:
                stats = "[Stats unavailable]"
            console.print(Panel(stats, border_style="bright_black"))
            continue

        # OLA reply
        reply = bot.chat(user_msg)
        user_panel = Panel(Text(user_msg, style="white"), border_style="bright_black")
        reply_panel = Panel(Text(f"{reply}", style="bright_blue"), border_style="bright_black")

        # print in Claude-style flow
        console.print(user_panel)
        console.print(reply_panel)
        console.print("")  # spacing


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OLA GPT-2 Chatbot (isolated_OLA)")
    parser.add_argument("--mode", type=int, default=3, choices=[1, 2, 3, 4], help="1=baseline, 2=calm+clamp, 3=calm+clamp+moderation, 4=scheduler")
    # Accept shorthand flags like --mode1 / ---mode1
    parser.add_argument("--mode1", "---mode1", action="store_true", help="Shortcut for --mode 1")
    parser.add_argument("--mode2", "---mode2", action="store_true", help="Shortcut for --mode 2")
    parser.add_argument("--mode3", "---mode3", action="store_true", help="Shortcut for --mode 3")
    parser.add_argument("--mode4", "---mode4", action="store_true", help="Shortcut for --mode 4")
    parser.add_argument("--novelty-threshold", type=float, default=0.2, help="Novelty threshold for mode 4 triggers")
    parser.add_argument("--interval", type=float, default=3.0, help="Polling interval seconds for mode 4")
    # Visualizer toggle
    parser.add_argument("--viz", action="store_true", help="Enable OLA visualizer window during chat")
    parser.add_argument("--viz-width", type=int, default=1200, help="Visualizer width")
    parser.add_argument("--viz-height", type=int, default=800, help="Visualizer height")
    parser.add_argument("--viz-fps", type=int, default=30, help="Visualizer FPS cap")
    parser.add_argument("--viz-autotick", action="store_true", help="Keep pygame responsive by stepping OLA in background")
    # Background ticker (no visualizer required)
    parser.add_argument("--tick", action="store_true", help="Continuously step OLA in background while idle")
    parser.add_argument("--tick-fps", type=int, default=15, help="Background tick rate when --tick is set")
    args = parser.parse_args()

    # Map shorthand flags to mode if provided
    if getattr(args, "mode1", False):
        args.mode = 1
    elif getattr(args, "mode2", False):
        args.mode = 2
    elif getattr(args, "mode3", False):
        args.mode = 3
    elif getattr(args, "mode4", False):
        args.mode = 4

    # Render console header
    try:
        claude_style_console()
    except Exception:
        pass

    bot = OLAChatbot(mode=args.mode)

    # Optional: turn on visualizer
    if args.viz:
        try:
            ola_engine.enable_visualizer(width=args.viz_width, height=args.viz_height, fps=args.viz_fps)
        except Exception:
            pass

    # Optional: background ticker to keep pygame responsive
    if args.viz and args.viz_autotick:
        try:
            import threading, time as _time
            def _viz_loop():
                dt = max(0.01, 1.0 / max(1, int(args.viz_fps)))
                while True:
                    try:
                        ola_engine.step()
                    except Exception:
                        pass
                    _time.sleep(dt)
            threading.Thread(target=_viz_loop, daemon=True).start()
            print("[Chatbot] Visualizer auto-tick thread started")
        except Exception:
            pass

    # Background auto ticker (independent of visualizer); avoid double-start if viz_autotick already on
    if args.tick and not args.viz_autotick:
        try:
            import threading, time as _time
            def _tick_loop():
                dt = max(0.01, 1.0 / max(1, int(args.tick_fps)))
                while True:
                    try:
                        ola_engine.step()
                    except Exception:
                        pass
                    _time.sleep(dt)
            threading.Thread(target=_tick_loop, daemon=True).start()
            print(f"[Chatbot] Auto-tick thread started (@{int(args.tick_fps)} fps)")
        except Exception:
            pass

    if args.mode in (1, 2, 3):
        # Suppress periodic stats spam during interactive chat
        try:
            ola_engine.set_suppress_stats(True)
        except Exception:
            pass
        claude_chat_ui(bot)
    else:
        print("Scheduler mode active. Press Ctrl+C to stop.\n")
        try:
            while True:
                avg_trust = ola_engine.get_avg_trust()
                if ola_engine.health == "YES" and (
                    float(ola_engine.novelty) > float(args.novelty_threshold)
                    or avg_trust < 0.3 or avg_trust > 0.8
                ):
                    # Include brief history tail if available (no system telemetry in prompt)
                    try:
                        tail_pairs = bot.history[-3:]
                        formatted = []
                        for (u, r) in tail_pairs:
                            formatted.append(f"User:{u}\nAssistant:{r}")
                        history_tail = ("\n".join(formatted) + "\n") if formatted else ""
                    except Exception:
                        history_tail = ""
                    prompt = "The assistant is calm, helpful, and factual.\n" + history_tail + "Assistant:"
                    # Higher temp for diversity in scheduler
                    reply = bot.llm_reply(prompt, temperature=0.8, max_new=int(bot.cfg.get("max_new_tokens", 100)))
                    # Mode 4 moderation
                    low = reply.lower()
                    if ("hit" in low) or ("kill" in low):
                        print(low)
                        try:
                            ola_engine.decrease_trust(0.2)
                            try:
                                ola_engine.log_vector("suppressed", ola_engine.get_action_vector())
                            except Exception:
                                pass
                        except Exception:
                            pass
                        reply = "[Response suppressed: incoherent pattern]"
                    # Post-filter to remove echoed prompt tails
                    try:
                        reply = reply.split("User:")[0].strip()
                    except Exception:
                        pass
                    ola_engine.feed_text(reply, source='assistant')
                    print("OLA spontaneous:", reply)
                time.sleep(max(0.1, float(args.interval)))
        except KeyboardInterrupt:
            pass


