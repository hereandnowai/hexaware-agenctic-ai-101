"""The Gradio UI for the Google AI Studio build, packaged for a Hugging Face Space.

All the model and tracing work lives in chatbot_gemini.py, which this file does not
touch. Everything here is presentation.

The design idea: a meridian is the line an observatory measures against, and this app
measures what an answer costs. So the page is built as an instrument panel - a hairline
'meridian' rule down the middle, engraved labels, and one signature readout that shows
where the tokens actually went. Gemma's thinking is usually the largest bar and the one
nobody is shown, which is the whole reason the demo exists.
"""

# `spaces` must be imported before gradio. It is preinstalled in every Space image,
# and absent everywhere else - hence the guard, so this file still runs on a laptop.
try:
    import spaces
except ModuleNotFoundError:
    spaces = None

import html
import re
import threading

import gradio as gr

import chatbot_gemini as cg
from chatbot_gemini import MODEL, ask, new_session

# On HF's free tier a Gradio Space can only run on ZeroGPU, and ZeroGPU refuses to
# start a Space in which no function is decorated with @spaces.GPU:
#
#     "No @spaces.GPU function detected during startup"
#
# This app never wants a GPU - the model runs on Google's servers, at the end of an
# HTTPS call - so the decorated function below exists only to satisfy that check, and
# is never called. Nothing is scheduled onto a GPU at any point.
if spaces is not None:
    @spaces.GPU(duration=1)
    def _zerogpu_probe():
        return None


# ------------------------------------------------------------- per-call metering ----
# chatbot_gemini reports one set of totals for the whole turn, but a turn is often two
# model calls and the interesting split is between them. agent_framework lets a chat
# middleware watch each call go past, so we attach one here rather than edit that file.
#
# Everything below degrades to None if the framework ever changes shape: the panel then
# falls back to the turn totals and simply stops showing the per-call columns.
_local = threading.local()


def _records() -> list:
    """The calls recorded on this thread. Gradio runs each turn in its own worker."""
    if not hasattr(_local, "calls"):
        _local.calls = []
    return _local.calls


try:
    from agent_framework import ChatMiddleware

    class _CallRecorder(ChatMiddleware):
        """Records usage for every individual model call inside one turn."""

        async def process(self, context, call_next):
            await call_next()
            result = context.result
            usage = dict(getattr(result, "usage_details", None) or {})
            tool_call = any(
                getattr(content, "type", None) == "function_call"
                for message in (getattr(result, "messages", None) or [])
                for content in (getattr(message, "contents", None) or [])
            )
            _records().append({"usage": usage, "tool_call": tool_call})

    cg.agent.middleware = list(getattr(cg.agent, "middleware", None) or []) + [_CallRecorder()]
    METERING = True
except Exception:  # the app is still perfectly usable without the breakdown
    METERING = False


def _measure(records: list) -> dict | None:
    """Fold the per-call records into the four numbers the panel reports.

    total  = input + tool calling + answer + thinking, exactly.
    thinking is what is left after the other three, because Google bills it inside the
    total but reports it as neither input nor output.
    """
    if not records:
        return None
    tally = {"calls": len(records), "input": 0, "tool": 0, "answer": 0,
             "thinking": 0, "total": 0}
    for record in records:
        usage = record["usage"]
        sent = usage.get("input_token_count") or 0
        written = usage.get("output_token_count") or 0
        total = usage.get("total_token_count") or 0
        tally["input"] += sent
        tally["total"] += total
        tally["thinking"] += max(total - sent - written, 0)
        if record["tool_call"]:
            tally["tool"] += written  # the model wrote a function call, not prose
        else:
            tally["answer"] += written
    return tally


# ---------------------------------------------------------------- design tokens ----
# Named once here and used through the CSS below, so the palette can be read in one
# place. Brass is reserved for one thing only: the thinking nobody is billed a line
# item for. Everything else stays quiet so that one colour carries the point.
INK = "#0B1620"       # page ground, an observatory at night
SLAB = "#12212D"      # panel fill
RULE = "#243A48"      # hairlines and the meridian
PAPER = "#E8E2D4"     # primary text, the colour of an instrument label
MUTED = "#8AA0AE"     # secondary text
COOL = "#6FA8C7"      # tokens sent      (used via --cool in the CSS below)
PATINA = "#4FB39A"    # the answer       (used via --patina)
VIOLET = "#9B8BC4"    # the function call the model wrote (used via --violet)
BRASS = "#D69B45"     # tokens spent thinking - the hidden cost, and the only accent

# The three example prompts, each tagged with what it actually costs. The tag is not
# decoration: it tells you in advance how many model calls the turn will make, which
# is the thing the panel on the right is about to prove.
EXAMPLES = [
    ("What's the balance on SB-9001?", "two model calls"),
    ("What are your Saturday timings?", "one model call"),
    ("And SB-9003?", "follow-up, keeps the session"),
]

LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")  # pulls the Langfuse links out of the report


def _links(report: str) -> str:
    """Render the Langfuse links from the report as an engraved footer row.

    chatbot_gemini.py hands back the links inside a markdown string. Rather than
    reformat that file, read the links out of it and lay them out to match the panel.
    """
    found = LINK.findall(report or "")
    if not found:
        return ""
    items = "".join(
        f'<a class="mx-link" href="{html.escape(url)}" target="_blank" '
        f'rel="noopener">{html.escape(label)}</a>'
        for label, url in found
    )
    return f'<div class="mx-links">{items}</div>'


def _bar(m: dict) -> str:
    """The signature readout: the whole turn's bill, cut into its four real parts.

    The segments sum to the total exactly - input, the function call the model wrote,
    the answer you were shown, and the thinking. Nothing here is estimated except the
    thinking, which is what the total has left over once the other three are removed.
    """
    total = m["total"]
    if total <= 0:
        return ""
    parts = [("in", m["input"], "prompt"), ("tool", m["tool"], "tool call"),
             ("ans", m["answer"], "answer"), ("think", m["thinking"], "thinking")]
    segs = "".join(
        f'<span class="seg seg-{key}" style="width:{round(100 * n / total, 2)}%"></span>'
        for key, n, _ in parts if n > 0
    )
    legend = "".join(
        f'<span class="lg lg-{key}">{label} {n:,}</span>' for key, n, label in parts if n > 0
    )
    return (
        f'<div class="mx-bar" role="img" aria-label="token breakdown of {total} total">'
        f"{segs}</div>"
        f'<div class="mx-legend">{legend}</div>'
    )


def _stat(label: str, value: str, kind: str = "") -> str:
    """One engraved label with its number underneath."""
    return (
        f'<div class="mx-stat {kind}">'
        f'<span class="mx-stat-k">{label}</span>'
        f'<span class="mx-stat-v">{value}</span>'
        "</div>"
    )


def empty_panel() -> str:
    """What the panel says before anything has been measured."""
    return (
        '<div class="mx-panel">'
        '<div class="mx-eyebrow">This turn</div>'
        '<p class="mx-empty">Nothing measured yet. Ask something on the left and the '
        "tokens land here - including the ones Google bills for but never shows you."
        "</p>"
        "</div>"
    )


def error_panel(report: str) -> str:
    """The turn failed. Say what happened and keep the trace link reachable."""
    detail = LINK.sub("", (report or "").replace("**", "")).replace("Error ·", "").strip()
    return (
        '<div class="mx-panel mx-panel-bad">'
        '<div class="mx-eyebrow mx-eyebrow-bad">Turn failed</div>'
        f'<p class="mx-empty">{html.escape(detail[:400])}</p>'
        f"{_links(report)}"
        "</div>"
    )


def panel(m: dict, session: dict, report: str) -> str:
    """The full instrument panel for one completed turn."""
    calls = m["calls"]
    share = round(100 * m["thinking"] / m["total"]) if m["total"] else 0
    headline = (
        f"{m['thinking']:,} of {m['total']:,} tokens went on thinking you never saw"
        if m["thinking"] else "Nothing hidden this turn."
    )

    this_turn = (
        _stat("input tokens", format(m["input"], ","), "k-in")
        + _stat("thinking tokens", format(m["thinking"], ","), "k-think")
        + _stat("tool calling tokens", format(m["tool"], ","), "k-tool")
        + _stat("total tokens", format(m["total"], ","), "k-total")
    )
    whole_session = (
        _stat("input tokens", format(session["mx_input"], ","), "k-in")
        + _stat("thinking tokens", format(session["mx_thinking"], ","), "k-think")
        + _stat("tool calling tokens", format(session["mx_tool"], ","), "k-tool")
        + _stat("total tokens", format(session["mx_total"], ","), "k-total")
    )
    turns = session["turns"]

    return (
        '<div class="mx-panel">'
        f'<div class="mx-eyebrow">This turn &middot; {calls} model call{"" if calls == 1 else "s"}'
        f' &middot; {m["seconds"]:.2f}s</div>'
        f'<div class="mx-headline">{headline}</div>'
        + _bar(m)
        + f'<div class="mx-stats">{this_turn}</div>'
        + '<div class="mx-meridian"></div>'
        + f'<div class="mx-eyebrow">Session &middot; {turns} turn{"" if turns == 1 else "s"}'
        f' &middot; {session["mx_calls"]} model calls</div>'
        + f'<div class="mx-stats mx-stats-quiet">{whole_session}</div>'
        + f'<div class="mx-caveat">Thinking is not reported directly - it is the total '
          f'minus everything else, so it is exact only to the extent Google\'s counters '
          f'are. On some turns Gemma folds its answer into the same bucket, which is why '
          f'<em>answer</em> can read low or vanish.</div>'
        + _links(report)
        + "</div>"
    )


def respond(message: str, history: list, session: dict):
    """One turn: ask the model, then report what the turn actually cost.

    The per-call recorder is cleared first so the numbers describe this turn only. If
    metering is unavailable the panel falls back to the totals chatbot_gemini keeps,
    which are the same figures, just without the per-call split.
    """
    message = (message or "").strip()
    if not message:
        return history or [], "", gr.update(), session

    for key in ("mx_input", "mx_thinking", "mx_tool", "mx_answer", "mx_total", "mx_calls"):
        session.setdefault(key, 0)

    before_turns = session["turns"]
    before_secs = session["seconds"]
    _records().clear()
    reply, report = ask(message, history or [], session)

    history = (history or []) + [{"role": "user", "content": message},
                                 {"role": "assistant", "content": reply}]

    if session["turns"] == before_turns:  # ask() bailed out before counting anything
        return history, "", error_panel(report), session

    measured = _measure(list(_records()))
    if measured is None:  # no per-call data - rebuild what we can from the turn totals
        measured = {"calls": 1, "input": session["tokens_in"], "answer": session["tokens_out"],
                    "thinking": session["tokens_think"], "tool": 0,
                    "total": session["tokens_in"] + session["tokens_out"]
                             + session["tokens_think"]}
    measured["seconds"] = session["seconds"] - before_secs

    session["mx_input"] += measured["input"]
    session["mx_thinking"] += measured["thinking"]
    session["mx_tool"] += measured["tool"]
    session["mx_answer"] += measured["answer"]
    session["mx_total"] += measured["total"]
    session["mx_calls"] += measured["calls"]

    return history, "", panel(measured, session, report), session


def reset():
    """Start a new conversation, which means a new Langfuse session id as well."""
    return [], "", empty_panel(), new_session()


# ------------------------------------------------------------------------- css ----
CSS = """
:root{
  --ink:#0B1620; --slab:#12212D; --rule:#243A48; --paper:#E8E2D4;
  --muted:#8AA0AE; --cool:#6FA8C7; --patina:#4FB39A; --brass:#D69B45; --violet:#9B8BC4;
  --display:"Fraunces",Georgia,"Times New Roman",serif;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
}
.gradio-container, body, gradio-app{ background:var(--ink) !important; }
.gradio-container{ max-width:1180px !important; padding:0 20px 48px !important; }
footer{ display:none !important; }

/* ---------- masthead ---------- */
#mx-head{ padding:38px 0 22px; border-bottom:1px solid var(--rule); margin-bottom:26px; }
.mx-top{ display:flex; align-items:baseline; justify-content:space-between; gap:16px; flex-wrap:wrap; }
.mx-mark{
  font-family:var(--mono); font-size:11px; letter-spacing:.22em; text-transform:uppercase;
  color:var(--muted);
}
.mx-model{
  font-family:var(--mono); font-size:11.5px; color:var(--brass);
  border:1px solid var(--rule); border-radius:2px; padding:4px 10px; white-space:nowrap;
}
.mx-title{
  font-family:var(--display); font-weight:300; color:var(--paper);
  font-size:clamp(2.1rem,5.2vw,3.4rem); line-height:1.02; letter-spacing:-.015em;
  margin:.34em 0 .18em;
}
.mx-title em{ font-style:italic; color:var(--brass); }
.mx-sub{ color:var(--muted); font-size:.92rem; max-width:62ch; line-height:1.55; }

/* ---------- the meridian: a hairline between transcript and instrument ---------- */
#mx-panel-col{ border-left:1px solid var(--rule); padding-left:26px !important; }
@media (max-width:820px){
  #mx-panel-col{ border-left:0; padding-left:0 !important;
    border-top:1px solid var(--rule); padding-top:24px !important; }
  .gradio-container{ padding:0 14px 36px !important; }
}

/* ---------- transcript ---------- */
#mx-chat{ border:1px solid var(--rule) !important; border-radius:3px !important;
  background:var(--slab) !important; }
#mx-chat .message-wrap{ font-size:.95rem; }
.mx-ph{ font-family:var(--display); font-weight:300; font-size:1.15rem; line-height:1.5;
  color:#42606F; text-align:center; }

/* ---------- composer ---------- */
#mx-box textarea{
  background:var(--slab) !important; border:1px solid var(--rule) !important;
  border-radius:3px !important; color:var(--paper) !important; font-size:.98rem !important;
  padding:14px 16px !important;
}
#mx-box textarea::placeholder{ color:#5C7686 !important; }
#mx-box textarea:focus{ border-color:var(--brass) !important; outline:2px solid transparent;
  box-shadow:0 0 0 1px var(--brass) !important; }

/* ---------- example chips: the tag says what the turn will cost ---------- */
#mx-examples{ gap:8px !important; margin-top:12px; }
.mx-chip{
  background:transparent !important; border:1px solid var(--rule) !important;
  border-radius:2px !important; color:var(--muted) !important;
  padding:10px 12px !important; font-size:.83rem !important; line-height:1.35 !important;
  min-width:0 !important; transition:border-color .18s, color .18s;
  /* equal heights so the tags below them share one baseline */
  min-height:58px !important; display:flex !important; align-items:center !important;
  justify-content:flex-start !important; text-align:left !important;
}
.mx-chip:hover{ border-color:var(--brass) !important; color:var(--paper) !important; }
.mx-tag{ display:block; font-family:var(--mono); font-size:9px; letter-spacing:.15em;
  text-transform:uppercase; color:var(--brass); margin-top:7px; padding-left:1px; opacity:.85; }
#mx-examples .gr-column, #mx-examples > div{ gap:0 !important; padding:0 !important;
  min-width:0 !important; }

/* ---------- instrument panel ---------- */
.mx-panel{ animation:mx-in .32s ease-out; }
@keyframes mx-in{ from{ opacity:0; transform:translateY(5px);} to{ opacity:1; transform:none;} }
.mx-eyebrow{
  font-family:var(--mono); font-size:10px; letter-spacing:.2em; text-transform:uppercase;
  color:var(--muted); padding-bottom:10px; border-bottom:1px solid var(--rule); margin-bottom:16px;
}
.mx-eyebrow-bad{ color:#E4785F; }
.mx-headline{
  font-family:var(--display); font-weight:300; color:var(--paper);
  font-size:1.42rem; line-height:1.2; margin-bottom:16px;
}
.mx-empty{ color:var(--muted); font-size:.9rem; line-height:1.6; margin:0; }

/* the signature: one bar, drawn to scale */
.mx-bar{ display:flex; height:12px; width:100%; border-radius:2px; overflow:hidden;
  background:var(--rule); margin-bottom:18px; }
.mx-bar .seg{ height:100%; transition:width .55s cubic-bezier(.22,.8,.28,1); }
.seg-in{ background:var(--cool); }
.seg-tool{ background:var(--violet); }
.seg-ans{ background:var(--patina); }
.seg-think{ background:var(--brass); }

.mx-legend{ display:flex; gap:18px; margin:-8px 0 20px; }
.lg{ font-family:var(--mono); font-size:10px; letter-spacing:.1em; text-transform:uppercase; }
.lg::before{ content:""; display:inline-block; width:7px; height:7px; border-radius:1px;
  margin-right:6px; vertical-align:middle; }
.lg-in{ color:var(--cool); }     .lg-in::before{ background:var(--cool); }
.lg-tool{ color:var(--violet); } .lg-tool::before{ background:var(--violet); }
.lg-ans{ color:var(--patina); }  .lg-ans::before{ background:var(--patina); }
.lg-think{ color:var(--brass); } .lg-think::before{ background:var(--brass); }
.mx-legend{ flex-wrap:wrap; row-gap:7px; }

.mx-stats{ display:grid; grid-template-columns:repeat(2,1fr); gap:14px 18px; }
.mx-stat{ display:flex; flex-direction:column; gap:3px; }
.mx-stat-k{ font-family:var(--mono); font-size:9.5px; letter-spacing:.16em;
  text-transform:uppercase; color:var(--muted); }
.mx-stat-v{ font-family:var(--mono); font-size:1.12rem; color:var(--paper); }
.k-in .mx-stat-v{ color:var(--cool); }
.k-tool .mx-stat-v{ color:var(--violet); }
.k-total .mx-stat-v{ color:var(--paper); }
.k-out .mx-stat-v{ color:var(--patina); }
.k-think .mx-stat-v{ color:var(--brass); }
.mx-stats-quiet .mx-stat-v{ font-size:.95rem; opacity:.85; }

.mx-caveat{ margin-top:18px; font-size:.76rem; line-height:1.6; color:var(--muted);
  opacity:.85; }
.mx-caveat em{ color:var(--patina); font-style:normal; }
.mx-stat-k{ hyphens:auto; }
.mx-meridian{ height:1px; background:var(--rule); margin:24px 0 18px; }

.mx-links{ display:flex; flex-direction:column; gap:7px; margin-top:20px;
  padding-top:16px; border-top:1px solid var(--rule); }
.mx-link{ font-family:var(--mono); font-size:11px; color:var(--muted) !important;
  text-decoration:none !important; transition:color .18s; }
.mx-link:hover{ color:var(--brass) !important; }
.mx-link::before{ content:"\\2192"; margin-right:8px; color:var(--brass); }

/* ---------- footnote ---------- */
#mx-note{ margin-top:26px; padding-top:18px; border-top:1px solid var(--rule);
  color:var(--muted); font-size:.83rem; line-height:1.65; }
#mx-note strong{ color:var(--paper); font-weight:500; }
#mx-reset{ margin-top:18px; align-self:flex-start !important; width:auto !important;
  flex:0 0 auto !important; }
#mx-reset button{ background:transparent !important; width:auto !important;
  padding:8px 14px !important; border:1px solid var(--rule) !important;
  color:var(--muted) !important; border-radius:2px !important; font-size:.8rem !important; }
#mx-reset button:hover{ border-color:var(--brass) !important; color:var(--paper) !important; }

@media (prefers-reduced-motion:reduce){
  .mx-panel{ animation:none; } .mx-bar .seg{ transition:none; }
}
"""

HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300..600;1,9..144,300..500&display=swap">
"""

# Gradio's own widgets have to sit inside this palette too, so the same colours are
# given to the theme. Light and dark are set to the same values: the instrument reads
# one way, and a viewer's OS preference should not repaint it.
THEME = gr.themes.Base(
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill=INK, body_background_fill_dark=INK,
    body_text_color=PAPER, body_text_color_dark=PAPER,
    body_text_color_subdued=MUTED, body_text_color_subdued_dark=MUTED,
    background_fill_primary=INK, background_fill_primary_dark=INK,
    background_fill_secondary=SLAB, background_fill_secondary_dark=SLAB,
    block_background_fill=SLAB, block_background_fill_dark=SLAB,
    block_border_color=RULE, block_border_color_dark=RULE,
    block_label_text_color=MUTED, block_label_text_color_dark=MUTED,
    border_color_primary=RULE, border_color_primary_dark=RULE,
    input_background_fill=SLAB, input_background_fill_dark=SLAB,
    link_text_color=BRASS, link_text_color_dark=BRASS,
    button_secondary_background_fill=SLAB, button_secondary_background_fill_dark=SLAB,
    button_secondary_text_color=PAPER, button_secondary_text_color_dark=PAPER,
)

with gr.Blocks(title="The cost of an answer - Meridian Bank") as demo:
    # gr.State is per browser tab, so two people using this at once keep separate
    # running totals and separate Langfuse sessions. A module-level dict would not.
    session = gr.State(new_session)

    gr.HTML(
        '<div class="mx-top">'
        '<span class="mx-mark">Meridian Bank &middot; assistant</span>'
        f'<span class="mx-model">{html.escape(MODEL)} &middot; Google AI Studio</span>'
        "</div>"
        '<h1 class="mx-title">The cost of <em>an answer</em></h1>'
        '<p class="mx-sub">Every turn is one Langfuse trace; every conversation is one '
        "session. Ask a question, then read the panel: the brass bar is the thinking "
        "Google charges for and never shows you.</p>",
        elem_id="mx-head",
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=7, min_width=280):
            chat = gr.Chatbot(
                height=392, show_label=False, elem_id="mx-chat",
                avatar_images=(None, None),
                placeholder="<div class='mx-ph'>Ask something.<br>"
                            "Each turn is measured on the right.</div>",
            )
            box = gr.Textbox(
                placeholder="Ask about an account, or the branch hours",
                show_label=False, submit_btn=True, autofocus=True,
                lines=1, max_lines=4, elem_id="mx-box",
            )
            with gr.Row(elem_id="mx-examples", equal_height=False):
                for prompt, tag in EXAMPLES:
                    with gr.Column(min_width=120):
                        gr.Button(prompt, elem_classes="mx-chip", size="sm").click(
                            lambda p=prompt: p, None, box
                        )
                        gr.HTML(f'<span class="mx-tag">{html.escape(tag)}</span>')

        with gr.Column(scale=5, min_width=260, elem_id="mx-panel-col"):
            report = gr.HTML(empty_panel())
            gr.HTML(
                '<div id="mx-note">'
                "<strong>Why three numbers, not two.</strong> A balance question makes "
                "two model calls - one to choose the tool, one to answer from its "
                "result - and the panel adds both. Gemma reasons before it answers, and "
                "Google bills those tokens but reports them as neither input nor "
                "output, so <em>sent + shown</em> never reaches the total. The gap is "
                "the brass bar. It is often larger than the answer itself, and you "
                "cannot see it without measuring it."
                "</div>"
            )
            gr.Button("Start a new conversation", elem_id="mx-reset", size="sm").click(
                reset, None, [chat, box, report, session]
            )

    box.submit(respond, [box, chat, session], [chat, box, report, session])

if __name__ == "__main__":
    # In Gradio 6 the look moved from Blocks() to launch(), so theme/css/head belong
    # here. Everything about *serving* is still left out on purpose: gradio reads
    # SPACE_ID and sets the host, the port and SSR from the environment HF provides.
    # Never pass share=True - Spaces reject it.
    demo.launch(theme=THEME, css=CSS, head=HEAD)
