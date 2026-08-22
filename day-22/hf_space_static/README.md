---
title: The Cost of an Answer
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
short_description: A bank assistant that shows what every answer costs in tokens
---

# The cost of an answer — static build

The same Meridian Bank teaching demo as the Gradio build, rebuilt as a **static** Space:
one HTML file, no server, no Python. It calls Google AI Studio straight from the
browser, runs the tool-calling loop in JavaScript, and reports what each turn cost.

Static Spaces are free on every account, which is the reason this version exists.

## Deploying it

1. **New Space** → SDK **Static** → it is free, no hardware choice to make
2. Upload **`index.html`** and **`README.md`** to the Space (that is the whole app)
3. Open it. The page asks for a Google AI Studio key on first use.

Get a key at <https://aistudio.google.com/apikey>.

## The one thing to explain to your class

**A static Space has no server, so it has no secrets.** There is no
`Settings → Variables and secrets` for a static Space, because there is no process to
read them. Anything written into `index.html` is downloaded by every visitor and
readable with View Source.

So this page never ships a key. It asks each visitor for their own, and keeps it in
that browser's `localStorage` — it is never sent anywhere except to Google, and never
committed to the repo. If a student ever hardcodes their key into the HTML to "make it
easier", they have published it, and they should rotate it at once.

That constraint is the lesson, not a workaround: it is exactly why the Gradio build
needs a server, and why real applications keep provider keys server-side.

## What is missing compared to the Gradio build

| | Gradio build | Static build |
|---|---|---|
| Runs the model | yes | yes, from the browser |
| Tool calling | yes | yes, in JavaScript |
| Token accounting | yes | yes, identical arithmetic |
| Langfuse tracing | yes | **no** — ingestion needs the secret key, which cannot be shipped |
| Key handling | Space secret, server-side | each visitor supplies their own |
| Cost | free only on ZeroGPU | free everywhere |

In place of Langfuse there is a **call inspector**: every model call the turn made, with
its own token counts, listed on the page. It shows the same thing a trace would — that
one turn is often two calls, and that the second prompt is bigger because it carries
the tool's result.

## Why the numbers are what they are

Google reports `prompt_tokens`, `completion_tokens` and `total_tokens`. The three do not
add up, because Gemma's thinking is billed inside the total but reported as neither. The
leftover is the thinking:

```
thinking = total_tokens - prompt_tokens - completion_tokens
```

The page splits `completion_tokens` further: on a call that returned a function call,
those tokens are the **tool call**; otherwise they are the **answer**. The four parts sum
to the total exactly.
