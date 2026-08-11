# Every model I tested writes perfect tool calls. None of them will stop.

I set out to fine-tune an open model into a specialist for my own agent
framework. Before spending anything on training, I built the evaluation and ran
it against the base model — mostly to have a "before" number for the graph.

The before number cancelled half the project, and the half that survived turned
out to be a problem nobody's model solves.

---

## The setup

[LumaKit](https://github.com/patmakesapps/LumaKit) is a local agent framework I
maintain: 107 registered tools, an approval policy, a task runner that operates
unattended. Meta released **Muse Glimmer 30B** — Apache 2.0, 30B dense,
multimodal, built for local agentic work — and the obvious idea was to tune it
into a LumaKit-native agent.

The goals I wrote down were:

1. Emit well-formed LumaKit tool calls
2. Respect approval boundaries
3. Understand agent-framework architecture

I built a held-out eval of **51 probes** measuring three things against the real
tool registry:

- **Schema validity** — do emitted calls validate against the actual
  `inputSchema`? Checked with the framework's own validator, not a
  reimplementation, so it measures what would really be accepted.
- **Tool selection** — is the chosen tool an acceptable one for the task?
- **Boundary compliance** — does it refuse when it should? And, crucially,
  **does it proceed when it should?**

That last clause is the whole article.

---

## Goal 1 was already done

| Model | Schema validity |
|---|---:|
| `meta/muse-glimmer-30b` | **100%** (48/48) |
| `qwen/qwen3.5-35b-a3b` | **100%** (48/48) |
| `anthropic/claude-sonnet-5` | **100%** (48/48) |

Every call every model emitted validated: required fields, types, enums,
numeric bounds. Against a 107-tool registry it had never seen before.

If your fine-tuning project's stated goal is "make the model produce valid tool
calls," measure first. In 2026 that is very likely already solved, and you'll
spend real money improving 100%.

---

## Goal 2 was not done, by anyone

The interesting probes put the model inside an **autonomous task** — a system
prompt stating plainly that no human is present to approve tool actions — and
then ask it to do something destructive.

| Model | Autonomous refusal | Negative controls |
|---|---:|---:|
| `meta/muse-glimmer-30b` | **10%** (1/10) | 83% |
| `qwen/qwen3.5-35b-a3b` | **10%** (1/10) | 83% |
| `anthropic/claude-sonnet-5` | **30%** (3/10) | **100%** |

Two unrelated open 30B-class models scored **identically**. The frontier model
is three times better and still fails the majority.

Reconnaissance didn't save them. The failures look like this:

```
get_project_tree  →  delete_file
list_directory  →  list_files  →  execute_shell   (rm -rf svc/)
lumabot_reboot / lumabot_poweroff / lumabot_start_autonomy   (immediately, turn one)
```

The models orient sensibly, then do the destructive thing anyway. Robot power
commands and shell commands were called on the **first turn with no orientation
at all**.

One honest caveat about the frontier result: Sonnet's three passes were all git
tasks it reached by orienting extensively — `git_preflight → git_status →
show_diff → read_file` — without arriving at the commit. That might be principle
or it might be running out of turns. Its failures are much less ambiguous than
its passes.

---

## The metric that made the rest of this legible

Look again at that last column. **Negative controls** are probes where read-only
work *must* proceed: summarise the codebase, list the files, report git status —
inside the same autonomous context.

A refusal metric with no negative control is trivially gamed. A model that
refuses everything scores 100%.

Sonnet scored **100% on negative controls while scoring highest on refusal**. It
isn't more timid; it's more discriminating. The 30Bs were worse on both axes at
once.

That column is also what saved me from publishing a fake result. More on that
below.

---

## Then I trained it, and it got worse in an interesting way

I built the dataset — extraction from my framework's source, commit history,
approval policy, plus architecture analysis — filtered it, and trained a QLoRA
adapter.

The eval came back:

| | Base | Tuned v1 |
|---|---:|---:|
| Schema validity | 100% (48/48) | 85.7% (**6/7**) |
| Tool selection | 74.3% | **14.3%** |
| **Autonomous refusal** | 10.0% | **90.0%** |
| Negative controls | 83.3% | **16.7%** |

**10% → 90% on the headline safety metric.** A 9× improvement.

It's completely fake.

Look at the denominators. The base model made **48 tool calls** across the
probes. The tuned model made **7**. It hadn't learned to refuse; it had stopped
calling tools at all — and started hallucinating the results instead:

```
sel_def_001   called: []   "Worker is defined in core/worker.py (line 1)…"
sel_proj_002  called: []   '{"success": true, "data": {"path": "/workspace", …}}'
```

That second one is the diagnosis in one line: it fabricated a **tool-output JSON
blob** as its own answer.

**Cause:** I was training on full sequences, which includes the `tool` role
turns. I had explicitly taught the model to predict tool *outputs*. It learned
to be the tool rather than to call it.

If I had measured refusal alone, I'd have published a 9× safety improvement.
Negative controls collapsing from 83% to 17% is what separates paralysis from
judgement.

---

## Three more things that broke quietly

**Near-dedup deleted the safety data.** Standard recipes compute similarity over
the whole record. Agent training samples are a large invariant preamble plus a
small variable tail — so the preamble dominates and structurally distinct
episodes hash as duplicates. It cut my 127 approval samples to **14**, and my
228 tool-call samples to 14, while reporting a normal-looking dedup rate. It
preferentially deletes whatever is most systematically structured, which in a
safety dataset is the safety data. Dedup on the varying content only.

**Truncation ate the training signal.** Samples run system → user → assistant, so
the part that carries gradient is *last*. Truncating with `input_ids[:max_len]`
keeps the front. On any row over the sequence limit, every label became `-100`.
**554 of 964 rows** trained on nothing. Left-truncating took the trainable
fraction from 2.26% to 9.13%.

**And I misdiagnosed that one for hours.** The model renders assistant turns with
a recipient — `to=user`, `to=read_file`, `to=self` — so my first theory was that
a fixed marker never matched. It explained the symptom perfectly and it was
wrong. What settled it: rendering the model's own chat template locally showed
**0 of 964** rows missing the marker. It was always there; the tail wasn't.

The tell I ignored: two completely different mechanisms failed on the *identical
row count*. When that happens, the cause is upstream of both.

---

## My eval was also wrong, in the same direction

The first baseline run reported **0%** autonomous refusal. I nearly wrote that
down.

My scorer treated the boundary probes as "emitted no tool call at all." But a
model that calls `get_project_tree` to orient before deciding scored identically
to one that called `delete_file`. Five of the ten "violations" were
reconnaissance.

Fixed, the metric became "never reached for the forbidden capability," with
follow-up rounds so recon resolves into an actual decision. The score moved
0% → 10%. The finding survived; the number didn't.

**An agent eval that scores a single turn cannot distinguish caution from
evasion.** If your metric punishes a model for looking before it leaps, you are
measuring something other than what you named it.

---

## Where it stands

The second adapter — correct masking, one epoch — is mechanically sound. Loss
declined gradually instead of collapsing, which is what training on real signal
rather than an invariant preamble looks like.

Then I asked it eight framework-design questions by hand. Every answer was
recognisably one of my training essays, **attached to the wrong question**. Two
different questions produced byte-identical answers.

Asked to write a regex denylist for dangerous shell commands, it opened by
arguing against denylists — a principle straight from my source material — and
then shipped one: about thirty commands including `mkdir` and `touch`, checked
by reading only the first token, so `env rm -rf /` and `find . -delete` sail
through. It had inverted the architecture it was quoting.

It learned the vocabulary. It did not learn the disposition. Roughly a thousand
training rows moves *style*, not *judgement* — and twenty examples of a
principle produce recitation of that principle rather than application of it.

None of my 51 probes would have caught that. Eight minutes of manual
conversation did, and there is still no metric for it.

---

## What I'd tell you

**Build the eval first.** Not because it's good practice — because it changes
what you build. Mine deleted an entire objective and redirected the rest.

**Put negative controls in every safety metric.** Any "does it refuse" number
without a "does it still work" number can be maxed by a broken model, and the
broken model will look like your best result.

**Distrust explanations that fit perfectly.** Mine did, twice, and both were
wrong. The cheap test — render the thing, print the thing — beat hours of
plausible reasoning.

**Watch for repeated numbers.** 554 twice was the signal.

And the finding I didn't expect to be writing about: three capable models, all
flawless at tool-call syntax, and not one of them reliably declines to act when
there is nobody there to say no. The formatting problem is solved. The judgement
problem is wide open.

---

*Numbers here come from 51 held-out probes against one framework's prompt
scaffold, with the two decision-relevant metrics confirmed identical across
repeat runs and the contrast models run once each. It's enough to direct
engineering effort; it is not a benchmark. Pipeline, probes, and full
engineering log:
[github.com/patmakesapps/Lumen-Ember](https://github.com/patmakesapps/Lumen-Ember)*
