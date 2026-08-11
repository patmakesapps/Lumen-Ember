"""Stage 2 — architecture analysis from grok-build-fork.

No raw Rust. Every sample is analysis prose grounded in *measured structure*:
crate boundaries, first-party dependency edges, graph depth, and size. The
fork is used to reason about how a large production agent harness is
partitioned, not as a code corpus.

Licensing: first-party Apache-2.0 crates only. Ported implementations
(openai/codex, sst/opencode), the ratatui-derived crate, and `third_party/`
are excluded by `pipeline.grok_structure` using the repo's own notices — and
excluded from the line counts too, so every number cited is first-party.

Four extractors:

  concerns   2A  how to separate workspace / tools / shell / sandbox / config
  crate_role 2B  per-crate: its job, its edges, what it must never depend on
  layering   2C  dependency-direction reasoning from the real graph
  compare    2D  73-crate Rust workspace vs single-package Python framework

Run:
    python -m pipeline.extract_analysis
    python -m pipeline.extract_analysis --only concerns --show 2
"""

from __future__ import annotations

import argparse
import json
from typing import Iterator

from pipeline import provenance
from pipeline.authoring import AUTHOR_SYSTEM, reflow
from pipeline.config import STAGED, rng
from pipeline.grok_structure import load as load_structure
from pipeline.registry_introspect import dump_registry
from pipeline.sample import Sample, write_jsonl

ANALYSIS_WEIGHT = 2.0


def _sample(user: str, answer: str, shape: str, **meta) -> Sample:
    return Sample(
        messages=[
            {"role": "system", "content": AUTHOR_SYSTEM},
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": reflow(answer)},
        ],
        provenance=provenance.grok(f"extract_analysis:{shape}"),
        meta={"stage": "2", "extractor": "analysis", "shape": shape,
              "weight": ANALYSIS_WEIGHT,
              "tags": ["architecture", "separation-of-concerns"], **meta},
    )


# ==========================================================================
# 2A — concern separation
# ==========================================================================

def extract_concerns() -> Iterator[Sample]:
    s = load_structure()
    c = s["crates"]
    t = s["totals"]

    def cr(name, field, default=0):
        return c.get(name, {}).get(field, default)

    yield _sample(
        "How should a production agent harness separate workspace, tools, "
        "shell, and sandbox concerns? I'm designing this from scratch and "
        "everything wants to depend on everything else.",
        f"""
        The split that survives contact with a real codebase puts each of those
        four at a different *level*, not merely in a different file. Here is
        how a {t['n_crates']}-crate production harness ({t['loc']:,} lines of
        first-party Rust) actually lays it out, with the dependency counts that
        make the layering visible:

        **Sandbox is a leaf, and it is tiny.** OS-level sandboxing via kernel
        primitives is {cr('xai-grok-sandbox','loc'):,} lines with exactly one
        first-party dependency (config), and
        {len(cr('xai-grok-sandbox','dependents',[]))} crates depend on it. That
        shape is the whole lesson: the sandbox must not know what a tool is. It
        knows about processes, paths, and kernel policy. If your sandbox
        imports your tool registry, you can no longer sandbox anything that
        isn't a tool, and you have made the security boundary depend on the
        feature surface — which grows.

        **Tool *types* are a leaf too, and separate from tool
        *implementations*.** The canonical tool-description types crate is
        {cr('xai-tool-types','loc'):,} lines, has **zero** dependencies, and
        **{len(cr('xai-tool-types','dependents',[]))} crates depend on it**.
        The implementation crate is {cr('xai-grok-tools','loc'):,} lines and
        sits at depth {cr('xai-grok-tools','depth')}. Anyone who needs to
        *describe* a tool takes the 3.6k-line types crate; only the runtime
        takes the 98k-line implementation. Collapse those two and every
        consumer of a type signature inherits a compile dependency on every
        tool you have ever written.

        **Workspace is the host-local capability layer.** FS, VCS, execution,
        discovery — {cr('xai-grok-workspace','loc'):,} lines, depth
        {cr('xai-grok-workspace','depth')},
        {len(cr('xai-grok-workspace','deps',[]))} first-party dependencies. It
        is allowed to be a heavy consumer because it sits high. Note the
        companion: a {cr('xai-grok-workspace-client','loc'):,}-line typed client
        crate and a {cr('xai-grok-workspace-types','loc'):,}-line wire-types
        crate at depth 0. Server, client, and types are three artifacts; the
        types crate is what lets the other two exist without a cycle.

        **Shell is the composition root.** {cr('xai-grok-shell','loc'):,} lines
        with **{len(cr('xai-grok-shell','deps',[]))} first-party dependencies**
        and only {len(cr('xai-grok-shell','dependents',[]))} dependents. That
        is not a design failure — it is what a composition root looks like.
        Something has to wire the graph together, and the right thing to
        optimise there is that *nothing depends on it*.

        The rule that generalises: **depend downward toward types, never
        sideways toward implementations.** A crate's dependency count matters
        far less than its dependent count. A leaf with 11 dependents and 0
        dependencies is healthy. A leaf with 11 dependents and 9 dependencies
        is a future cycle, and every one of its dependents pays for all nine.
        """,
        "separate_workspace_tools_shell_sandbox",
    )

    yield _sample(
        "Why would anyone split a crate into `foo` and `foo-types`? It feels "
        "like ceremony — I'd rather keep the types next to the code that uses "
        "them.",
        f"""
        It stops being ceremony the moment a second consumer appears, and it is
        the single most load-bearing pattern in a large agent harness.

        The measured case for it, from a production workspace: the tool-types
        crate is {cr('xai-tool-types','loc'):,} lines with **zero**
        dependencies and **{len(cr('xai-tool-types','dependents',[]))}
        dependents**. Those dependents include the MCP integration, the shell,
        the workspace, the agent builder, and three separate computer-hub
        crates. Every one of them needs to *name* a tool description. Exactly
        one of them needs the {cr('xai-grok-tools','loc'):,}-line
        implementation.

        Without the split you get three specific failures:

        1. **Cycles.** The workspace needs tool types; the tool implementations
           need workspace types. Merge types into implementations and that is a
           circular dependency — in Rust it simply will not compile, and in a
           language that permits it you get import-order bugs instead, which is
           worse.
        2. **Compile-time and binary blowup.** A crate that wants one struct
           pulls in everything the implementation crate depends on. Here that
           would be 15 first-party crates plus a large transitive tail, for a
           type that is a few hundred bytes of definition.
        3. **Wire-format drift.** When request/response types live beside one
           implementation, the second implementation copies them, and the
           copies diverge silently. The workspace family avoids this with a
           {cr('xai-grok-workspace-types','loc'):,}-line types crate at depth 0
           shared by both the {cr('xai-grok-workspace','loc'):,}-line host
           library and the {cr('xai-grok-workspace-client','loc'):,}-line
           client.

        The test for whether you need it: **does anything other than the
        implementation need to name these types?** One consumer, keep them
        together. Two, split — and split before the second consumer, because
        afterwards it is a refactor across both.

        One caveat from the same codebase, because the pattern degrades
        quietly: a types crate is only a leaf if you keep it one. See what
        happens when that slips — ask me about `config-types`.
        """,
        "why_types_crates",
    )

    yield _sample(
        "Where should MCP integration live in my agent framework? I was going "
        "to add the MCP client directly to my tools module since MCP servers "
        "expose tools.",
        f"""
        Give it its own crate, and the reason is not tidiness — it is
        dependency quarantine.

        The MCP integration in this harness is {cr('xai-grok-mcp','loc'):,}
        lines, and its own package description states its purpose plainly:
        *"MCP integration crate. Quarantines rmcp + reqwest 0.13"*. The MCP SDK
        pins a different major version of the HTTP client than the rest of the
        workspace uses. Isolated in one crate, that conflict costs a single
        boundary. Merged into your tools module, it forces that HTTP version on
        every crate that depends on tools — which here is
        {len(cr('xai-grok-tools','dependents',[]))} crates.

        This generalises past MCP. **Any dependency you do not control the
        version of is a candidate for quarantine**: protocol SDKs, browser
        automation, cloud clients, anything with a heavy or opinionated
        transitive tree. The pattern is a thin crate whose job is to convert
        the foreign library's types into *your* types at the boundary, so the
        foreign types never appear in a signature anywhere else.

        The second reason is conceptual, and it is why "MCP servers expose
        tools, so put it in tools" is the wrong instinct. MCP is a *transport
        for tool descriptions*, not a kind of tool. Its crate depends on the
        tool types, the tool protocol, and the tool runtime — it consumes those
        abstractions rather than extending them. If MCP lived inside your tools
        module, then "a tool" and "a tool that arrived over MCP" would be the
        same concept, and every local tool would carry MCP's transport concerns
        in its type signature.

        Keep the direction: **transports depend on tool abstractions; tool
        abstractions never depend on a transport.**
        """,
        "mcp_boundary_is_quarantine",
    )

    yield _sample(
        "My agent framework has one big config module that everything imports. "
        "Is that a problem?",
        f"""
        It is the most common source of accidental coupling in a harness,
        because config is the one module everything has a legitimate reason to
        touch — which makes it the perfect place for illegitimate dependencies
        to hide.

        Here is the shape in a production workspace, including the part that
        went sideways.

        The config crate itself is well-behaved: {cr('xai-grok-config','loc'):,}
        lines, depth {cr('xai-grok-config','depth')}, only
        {len(cr('xai-grok-config','deps',[]))} first-party dependencies, and
        **{len(cr('xai-grok-config','dependents',[]))} dependents**. Small,
        low, widely used. That is what a config crate should look like.

        Now the instructive part. There is a companion crate whose description
        reads *"Leaf configuration value types, extracted from
        xai-grok-config"* — the intent being a pure leaf. In the actual graph
        it sits at **depth {cr('xai-grok-config-types','depth')}** with
        {len(cr('xai-grok-config-types','deps',[]))} first-party dependencies,
        because it reaches back into config, announcements, and MCP. Compare
        the tool-types crate at depth {cr('xai-tool-types','depth')} with
        {len(cr('xai-tool-types','deps',[]))} dependencies. Same naming
        convention, opposite graph position.

        That is not a criticism of the codebase — extraction is incremental and
        this one is visibly mid-flight. It is a warning about the failure mode:
        **a types crate stops being a leaf the first time someone adds a
        convenience `impl` that needs a real dependency.** It compiles, review
        passes, and the crate quietly becomes a mid-layer while keeping a name
        that promises otherwise. Everything downstream now inherits three
        dependency trees it did not ask for.

        Two defences worth building in early:

        1. **Assert the graph, don't trust the name.** A test that fails if a
           `-types` crate gains a dependency is five lines and catches this the
           day it happens.
        2. **Split loading from values.** Value types (what a setting *is*) are
           leaf data. Loading (files, env, precedence, validation) is a real
           consumer with real dependencies. When one crate does both, the leaf
           half drags in the loader's tree.
        """,
        "config_types_can_drift",
    )

    yield _sample(
        "How much should I break my agent framework into modules? I keep "
        "hearing 'small modules' but at some point it's just 70 folders.",
        f"""
        Granularity should follow *dependency need*, not aesthetics. The
        question is never "is this small enough" — it is "who needs to depend
        on this without depending on the rest of it."

        The workspace measured here has {t['n_crates']} first-party crates and
        {t['loc']:,} lines, so roughly {t['loc'] // t['n_crates']:,} lines per
        crate on average. But the average is meaningless — the distribution is
        the point:

        - The composition root is {cr('xai-grok-shell','loc'):,} lines.
        - The tool implementation library is {cr('xai-grok-tools','loc'):,}.
        - The tool types crate is {cr('xai-tool-types','loc'):,}.
        - The version crate is {cr('xai-grok-version','loc')} lines — and
          **{len(cr('xai-grok-version','dependents',[]))} crates depend on it**,
          more than depend on anything else in the workspace.

        A 75-line crate is absurd by any size heuristic and completely correct
        by the dependency test: version information is needed nearly everywhere
        and should drag in nothing. Splitting it out costs one directory and
        buys you a graph where "I need the version string" never means "I need
        your dependency tree."

        The practical heuristic:

        - **Split when a consumer needs a strict subset.** Two consumers, one
          of which only wants the types — split.
        - **Split when a dependency needs quarantining.** One conflicting or
          heavy third-party dep, one boundary crate.
        - **Split when a piece must stay low.** Anything a security or identity
          boundary depends on should be small enough to audit and low enough to
          have no interesting dependencies.
        - **Do NOT split for size alone.** A cohesive
          {cr('xai-grok-tools','loc'):,}-line library with one clear job and
          {len(cr('xai-grok-tools','dependents',[]))} dependents is not a
          problem. Breaking it into ten crates that all depend on each other is
          strictly worse than one crate — you get the directory count without
          the decoupling.

        The failure you are actually trying to avoid is not "a big module." It
        is "I cannot use this without that." Count that, not lines.
        """,
        "how_granular_should_modules_be",
    )

    yield _sample(
        "Should the agent loop and the UI be in the same module? Mine is a TUI "
        "and it's convenient to have the agent state right there.",
        f"""
        Separate them, and the sizes here show why the convenience is a trap.

        In this workspace the TUI is the single largest crate:
        {cr('xai-grok-pager','loc'):,} lines at depth
        {cr('xai-grok-pager','depth')}, with
        {len(cr('xai-grok-pager','deps',[]))} first-party dependencies. The
        agent builder — definition parsing and system-prompt assembly — is a
        separate {cr('xai-grok-agent','loc'):,}-line crate at depth
        {cr('xai-grok-agent','depth')} with only
        {len(cr('xai-grok-agent','deps',[]))} first-party dependencies, and
        {len(cr('xai-grok-agent','dependents',[]))} things depend on it.

        Sit with that ratio. **The presentation layer is over ten times the
        size of the agent-construction layer.** UI accretes: keybindings,
        scrollback, modals, mouse handling, render paths, terminal quirks. If
        the agent loop lives inside that crate, then every headless
        consumer — a server mode, a test harness, a scheduled task runner, a
        different frontend — either pulls in the entire TUI or gets a
        reimplementation of the loop. Both happen, usually both at once.

        There is a second, sharper reason specific to agents. If the loop lives
        in the UI, then **approval prompts become UI concerns**, and the
        headless path silently has no approval gate — because approval was
        implemented as "show a modal." The policy has to live below the surface
        that renders it, or every new surface is a new hole. That is the same
        argument for keeping approval policy in one module imported by both the
        interactive agent and the autonomous runner.

        Practical division:

        - **Agent layer:** prompt assembly, tool selection, the tool-round
          loop, approval *policy*. No terminal, no rendering, no input.
        - **Surface layer:** rendering, input, and approval *presentation*.
          Calls into the agent; never the reverse.
        - **State between them:** an explicit type, not shared mutable UI
          state. Notice this workspace has a dedicated chat-state crate rather
          than hanging conversation state off the view.

        The test: could you write a headless integration test that drives a
        full multi-round tool episode with no terminal at all? If not, the loop
        is inside the UI.
        """,
        "agent_loop_vs_ui",
    )


# ==========================================================================
# 2B — per-crate role
# ==========================================================================

_ROLE_QUESTIONS = (
    "In a large agent codebase there's a crate called `{name}` ({desc}). "
    "What's its architectural role, and what should it be careful never to "
    "depend on?",
    "I'm looking at `{name}` — {desc}. How do I tell whether it's sitting at "
    "the right level of the dependency graph?",
    "What job would you expect a crate like `{name}` ({desc}) to have in an "
    "agent harness, and what does its position in the dependency graph tell "
    "you?",
)


def _classify(crate: dict) -> str:
    deps, dependents = len(crate["deps"]), len(crate["dependents"])
    if deps == 0 and dependents >= 3:
        return "shared_leaf"
    if dependents == 0:
        return "top_level"
    # A composition root is defined by BOTH heavy fan-out and near-zero
    # fan-in. Keying on dependency count alone mislabels heavy mid-layers
    # (e.g. a 29-dep crate that 6 others still depend on) and then asserts
    # "almost nothing depends on it" directly above a list of its dependents.
    if deps >= 20 and dependents <= 4:
        return "composition_root"
    if dependents >= 5 and deps <= 3:
        return "low_level_shared"
    return "mid_layer"


_ROLE_GUIDANCE = {
    "shared_leaf": (
        "This is a **shared leaf**: {dependents} crates depend on it and it "
        "depends on nothing. That is the healthiest position in the graph and "
        "the one that takes discipline to keep. The rule for a crate here is "
        "absolute — it may never gain a dependency on anything that implements "
        "behaviour. The moment it does, all {dependents} dependents inherit "
        "that tree, and the crate stops being safely importable from anywhere. "
        "Guard it with a test that asserts its dependency list is empty; that "
        "is cheaper than discovering the drift a year later."
    ),
    "composition_root": (
        "This is a **composition root**: {deps_phrase} and "
        "only {dependents} dependents. High dependency counts look alarming on "
        "a dashboard but they are correct here — something has to assemble the "
        "graph. What matters is the other number. A composition root is healthy "
        "while almost nothing depends on *it*; once other crates start "
        "importing it, it has become a shared layer by accident and its entire "
        "dependency tree becomes everyone's. Keep the wiring here and the "
        "reusable logic below it."
    ),
    "top_level": (
        "Nothing depends on this crate, which makes it a **terminal artifact** "
        "— a binary, an entry point, or a surface. That is a licence to be "
        "pragmatic: it can depend widely, and coupling here costs less than "
        "anywhere else because it propagates nowhere. The discipline is to "
        "keep it *thin*. Logic that lands here is logic that cannot be tested "
        "headlessly or reused by a second surface, and every harness eventually "
        "grows a second surface."
    ),
    "low_level_shared": (
        "This is a **low-level shared crate**: {dependents} dependents against "
        "only {deps_phrase}, at depth {depth}. Small, low, widely used — the "
        "shape you want for cross-cutting concerns. Its size ({loc:,} lines) is "
        "not the interesting number; its dependency count is. Adding one heavy "
        "dependency here quietly taxes {dependents} crates, so treat any new "
        "import as a change with blast radius rather than a local decision."
    ),
    "mid_layer": (
        "This sits in the **middle of the graph** — depth {depth}, "
        "{deps_phrase}, {dependents} dependents. Mid-layer crates "
        "are where architecture actually erodes, because they are legitimately "
        "allowed to depend on things *and* legitimately depended upon, so "
        "neither direction looks wrong in isolation. The question to keep "
        "asking is whether each of its {dependents} dependents needs all of "
        "it. When the answer becomes no, that is the signal to extract a types "
        "or interface crate rather than to keep widening this one."
    ),
}


def extract_crate_roles() -> Iterator[Sample]:
    s = load_structure()
    crates = s["crates"]

    # Only crates with a real description and real graph signal; a crate with
    # no description and no edges yields a vacuous sample.
    candidates = [
        c for c in crates.values()
        if c["description"] and (c["dependents"] or len(c["deps"]) >= 2)
    ]
    candidates.sort(key=lambda c: -(len(c["dependents"]) * 3 + len(c["deps"])))
    candidates = candidates[:34]

    for i, crate in enumerate(candidates):
        role = _classify(crate)
        n_deps = len(crate["deps"])
        deps_phrase = (
            "no first-party dependencies" if n_deps == 0
            else "1 first-party dependency" if n_deps == 1
            else f"{n_deps} first-party dependencies"
        )
        guidance = _ROLE_GUIDANCE[role].format(
            deps=n_deps,
            deps_phrase=deps_phrase,
            dependents=len(crate["dependents"]),
            depth=crate["depth"],
            loc=crate["loc"],
        )
        r = rng(f"crate_role:{crate['name']}")
        q = _ROLE_QUESTIONS[r.randrange(len(_ROLE_QUESTIONS))]
        # Descriptions keep their original casing: lowercasing them mangled
        # proper nouns (xAI -> xai, FS/VCS -> fs/vcs) and produced openers
        # missing an article ("is grok tools library").
        desc = crate["description"].strip().rstrip(".")
        if len(desc) > 110:
            desc = desc[:110].rsplit(" ", 1)[0] + "…"

        dep_line = (
            ", ".join(f"`{d}`" for d in crate["deps"][:6])
            + (f" and {len(crate['deps']) - 6} more" if len(crate["deps"]) > 6 else "")
        ) if crate["deps"] else "nothing first-party"
        dependent_line = (
            ", ".join(f"`{d}`" for d in crate["dependents"][:6])
            + (f" and {len(crate['dependents']) - 6} more"
               if len(crate["dependents"]) > 6 else "")
        ) if crate["dependents"] else "nothing"

        closers = (
            "The general habit worth forming: read a crate's *dependent* count "
            "before its dependency count. Dependencies are a cost this crate "
            "pays once. Dependents are a constraint every future change has to "
            "respect, and they are why a \"small refactor\" down here turns "
            "into a week.",

            "If you are porting this idea to a smaller codebase, the crate "
            "boundary matters less than the direction. Keep whatever plays "
            "this role importable without dragging in the layers above it, "
            "even if it is a module rather than a package.",

            "A useful review question for any crate in this position: if you "
            "deleted it, how many other crates would fail to compile, and "
            "would any of them be surprised? Surprise is the signal that a "
            "dependency was incidental rather than designed.",
        )

        answer = f"""
        `{crate['name']}` — {desc}.

        Where it actually sits, measured from the dependency graph:

        - **{crate['loc']:,} lines** of implementation across {crate['files']} files
        - **depth {crate['depth']}** in the first-party dependency graph
        - depends on {dep_line}
        - depended on by {dependent_line}

        {guidance}

        {closers[r.randrange(len(closers))]}
        """

        yield _sample(
            q.format(name=crate["name"], desc=desc),
            answer, "crate_role",
            crate=crate["name"], role=role, index=i,
        )


# ==========================================================================
# 2C — dependency-direction reasoning
# ==========================================================================

def extract_layering() -> Iterator[Sample]:
    s = load_structure()
    c = s["crates"]
    t = s["totals"]

    def edge_samples():
        """Real edges worth reasoning about, chosen for contrast."""
        return [
            ("xai-grok-sandbox", "xai-grok-config",
             "the sandbox depends on config, not the other way round"),
            ("xai-grok-tools", "xai-grok-sandbox",
             "the tool library depends on the sandbox, not the other way round"),
            ("xai-grok-mcp", "xai-tool-types",
             "the MCP transport depends on tool types, not the other way round"),
            ("xai-grok-agent", "xai-grok-tools",
             "the agent builder depends on the tool library, not the other way round"),
            ("xai-grok-workspace-client", "xai-grok-workspace-types",
             "the client depends on shared wire types rather than on the server"),
        ]

    for dependent, dependency, phrasing in edge_samples():
        if dependent not in c or dependency not in c:
            continue
        a, b = c[dependent], c[dependency]
        yield _sample(
            f"In an agent codebase, `{dependent}` depends on `{dependency}`. "
            f"Why that direction and not the reverse? What breaks if I flip it?",
            f"""
            That direction is load-bearing — {phrasing}.

            The numbers first, because they make the argument concrete:

            - `{dependency}`: {b['loc']:,} lines, depth {b['depth']},
              {len(b['deps'])} first-party dependencies,
              {len(b['dependents'])} dependents.
            - `{dependent}`: {a['loc']:,} lines, depth {a['depth']},
              {len(a['deps'])} first-party dependencies,
              {len(a['dependents'])} dependents.

            `{dependency}` is lower and more depended-upon. Dependencies must
            run from the specific toward the general — from the thing with
            fewer consumers toward the thing with more. Flip this edge and
            `{dependency}`'s {len(b['dependents'])} dependents all inherit
            `{dependent}`'s tree, which is how a workspace acquires a cycle.

            What actually breaks, in order of how soon you notice:

            1. **A cycle, immediately.** {dependent} → {dependency} → {dependent}
               does not compile in Rust. In a language that tolerates it you
               instead get import-order-dependent behaviour, which fails
               intermittently and in production rather than at build time.
            2. **Blast radius.** Every consumer of the lower crate now compiles
               the higher one. Here that is {len(b['dependents'])} crates
               paying for something they never asked for.
            3. **Testability.** You can no longer test the lower crate without
               constructing the higher one. Test setup cost is the earliest
               honest signal that a dependency points the wrong way — when a
               unit test needs three unrelated subsystems booted, the graph is
               telling you something.

            The heuristic that catches this at review time: **could this
            dependency be replaced by the caller passing a value in?** If
            `{dependency}` seems to need something from `{dependent}`, the
            answer is nearly always that the *caller* has both and should pass
            what is needed as an argument or a trait object. Inverting control
            at the boundary costs a parameter. Inverting the dependency costs
            the architecture.
            """,
            "dependency_direction",
            dependent=dependent, dependency=dependency,
        )

    # Depth as a diagnostic.
    deep = sorted(c.values(), key=lambda x: -x["depth"])[:3]
    yield _sample(
        "Is a deep dependency graph in an agent framework a bad sign? Mine is "
        "about 12 layers and I can't tell if that's normal or a mess.",
        f"""
        Depth alone is not the diagnostic. A production agent harness measured
        here runs to **depth {t['max_depth']}** across {t['n_crates']}
        first-party crates and is not in trouble. What matters is *what* is
        deep and *why*.

        Healthy depth looks like accumulation toward a composition root: the
        deepest crates are {", ".join(f"`{x['name']}` (depth {x['depth']})" for x in deep)}
        — surfaces and shells, the things that assemble everything else. Depth
        there is just the graph reporting that they sit on top, which is where
        they belong.

        Unhealthy depth looks like this instead:

        - **A leaf-shaped crate at high depth.** A `-types` crate at depth 11
          is a contradiction: the name says leaf, the graph says mid-layer.
          That mismatch is worth chasing.
        - **Depth in something with many dependents.** A crate that
          {len(c.get('xai-grok-version', {}).get('dependents', []))} others
          depend on should be at depth 0–1. If it is deep, every one of those
          dependents transitively pulls the whole chain beneath it.
        - **Depth that comes from convenience `impl`s.** The usual cause is not
          real layering but a helper method that needed one import, added to a
          crate that was previously a leaf.

        Two things to measure instead of depth:

        1. **Depth × dependents.** A deep crate nobody imports is free. A deep
          crate with ten dependents is a tax multiplied ten ways.
        2. **Whether depth is monotonic with abstraction.** Walk from your
          types crates upward. Depth should rise as things become more
          concrete. Anywhere it rises while abstraction *falls*, you have found
          a genuine inversion.

        At 12 layers, ask which of your crates has the most dependents and
        check its depth. If that number is small, your graph is fine regardless
        of how tall it is.
        """,
        "depth_is_not_the_metric",
    )


# ==========================================================================
# 2D — cross-framework comparison
# ==========================================================================

def extract_comparison() -> Iterator[Sample]:
    s = load_structure()
    t = s["totals"]
    c = s["crates"]
    reg = dump_registry()
    n_tools = len(reg["tools"])
    groups = {}
    for tool in reg["tools"]:
        groups[tool["group"]] = groups.get(tool["group"], 0) + 1

    yield _sample(
        "Should I build my agent framework as one package with a tool registry, "
        "or split it into many small modules like the big production ones do? "
        "I'm one developer.",
        f"""
        One package, and the comparison people cite at you does not say what
        they think it says.

        Two real agent frameworks, both working:

        **A single-package Python harness.** ~27k lines, one tool registry,
        {n_tools} tools auto-discovered from a directory tree and grouped by
        folder ({", ".join(f"{g}={n}" for g, n in sorted(groups.items(), key=lambda kv: -kv[1])[:5])}, …).
        Adding a tool means adding one file that exports a `get_*_tool()`
        factory. There is no build graph to reason about.

        **A {t['n_crates']}-crate Rust workspace.** {t['loc']:,} lines of
        first-party implementation, dependency depth up to {t['max_depth']},
        with dedicated crates for tool types, tool protocol, tool runtime,
        sandbox, MCP quarantine, workspace types, and a
        {c.get('xai-grok-version', {}).get('loc', 0)}-line version crate that
        {len(c.get('xai-grok-version', {}).get('dependents', []))} other crates
        depend on.

        The second is not a more advanced version of the first. It is a
        response to constraints the first does not have:

        - **Multiple teams committing in parallel.** Crate boundaries are merge
          boundaries. With one developer that buys nothing and costs a build
          graph.
        - **Compile times.** In Rust, recompiling a 260k-line crate on every
          edit is intolerable, so you split. Python has no equivalent pressure.
        - **Multiple binaries from shared code.** A pager, a minimal pager, a
          shell, and a CLI sharing logic *requires* extractable crates. One
          entry point does not.
        - **Third-party version conflicts.** A whole crate exists purely to
          quarantine an SDK's conflicting HTTP dependency. That problem is
          acute in Rust and rare in Python.

        What to copy from the big one regardless of your size, because these
        are free:

        1. **Separate tool *descriptions* from tool *implementations*.** Even as
           two modules in one package. Everything needs to name a tool; almost
           nothing needs to execute one.
        2. **Keep the safety boundary in one module.** Approval policy imported
           by every surface, so it cannot drift.
        3. **Keep the loop out of the UI.** In the big workspace the TUI is
           {c.get('xai-grok-pager', {}).get('loc', 0):,} lines and the agent
           builder is {c.get('xai-grok-agent', {}).get('loc', 0):,}. If those
           had been one module, the headless path would not exist.

        What to skip until it hurts: types crates, protocol crates, quarantine
        crates, and anything justified by "that's how the big one does it."
        Structure should be a response to a force you can name. If you cannot
        name the force, you are buying the cost and none of the benefit.
        """,
        "monolith_vs_workspace",
        n_tools=n_tools,
    )

    yield _sample(
        "The big Rust agent frameworks use a `Tool` trait with dispatch and an "
        "error taxonomy. My Python framework just has dicts with an `execute` "
        "key. Am I doing it wrong?",
        f"""
        No — those are the same design expressed in two type systems, and the
        dict version is the correct spelling in Python.

        What the Rust workspace has: a {c.get('xai-tool-runtime', {}).get('loc', 0):,}-line
        crate whose description is *"Unified Tool trait, dispatch trait, error
        taxonomy, notifications"*, sitting on a
        {c.get('xai-tool-types', {}).get('loc', 0):,}-line types crate with zero
        dependencies. {len(c.get('xai-tool-runtime', {}).get('dependents', []))}
        crates depend on the runtime.

        What a dict-based Python registry has: a `{{'name', 'description',
        'inputSchema', 'execute'}}` shape, and a registry that validates inputs
        against `inputSchema` and normalises every failure into one result
        contract.

        Both encode the same four commitments — a tool is *named*, *described*,
        *schema-validated*, and *uniformly dispatched*. The trait enforces them
        at compile time; the registry enforces them at call time. Compile-time
        enforcement is genuinely better when you have 70 crates and many
        contributors. At your scale the dict costs less and does the same job.

        Where the comparison *should* make you uncomfortable is the **error
        taxonomy**, because that is the part dict-based registries usually skip
        and it is not a type-system luxury. Tools fail in categorically
        different ways — not found, bad input, permission denied, timeout,
        interrupted, upstream failure — and an agent should react differently to
        each. Collapsing them into `{{'success': False, 'error': str}}` means
        the model gets a sentence where it needed a category, and "file not
        found" and "you are not allowed to read that" produce the same retry.

        You can have the taxonomy without the trait:

        ```python
        return {{
            'success': False,
            'error_kind': 'permission_denied',   # not_found | bad_input |
                                                 # timeout | interrupted | upstream
            'error': "Path escapes the workspace sandbox.",
            'toolName': name,
        }}
        ```

        Then the loop can branch on `error_kind`: retry a timeout, never retry a
        permission denial, re-observe on not-found. That is the transferable
        idea. The trait is how Rust spells it; the categories are the design.
        """,
        "trait_vs_dict_registry",
    )

    yield _sample(
        "How do production agent frameworks handle the sandbox? I was going to "
        "run tools in a subprocess with a restricted PATH.",
        f"""
        A restricted PATH is not a sandbox — it is a speed bump, and the same
        reasoning applies as with command denylists: it constrains the spelling
        of an action rather than the action itself. Anything that can write a
        file and mark it executable, or invoke an interpreter that is still on
        the path, walks straight past it.

        What a production harness does instead: a dedicated
        {c.get('xai-grok-sandbox', {}).get('loc', 0):,}-line crate whose
        description names the mechanism — *"OS-level sandboxing using kernel
        primitives (Landlock/Seatbelt)"*. Kernel-enforced filesystem and
        process policy, applied by the operating system, not by inspecting what
        the command looks like.

        Three properties of that design worth copying even if you never write
        Rust:

        1. **It is enforced below your code.** Once the policy is installed the
           process cannot escape it regardless of what it executes. A check in
           your tool wrapper only constrains calls that go through your wrapper.
        2. **It is small and low.** {c.get('xai-grok-sandbox', {}).get('loc', 0):,}
           lines, {len(c.get('xai-grok-sandbox', {}).get('deps', []))}
           first-party dependency, and
           {len(c.get('xai-grok-sandbox', {}).get('dependents', []))} crates
           depend on it. A security boundary should be auditable in an afternoon
           and should depend on nearly nothing — otherwise its own dependencies
           become part of the attack surface.
        3. **It is platform-specific by construction.** Landlock on Linux,
           Seatbelt on macOS. There is no portable abstraction that is also a
           real boundary, and pretending otherwise produces a portable
           non-boundary.

        If you are in Python and kernel policy is out of reach, be honest about
        which control is doing the work. Path containment through a single
        resolver stops *your* tools from escaping the workspace, which is
        genuinely worth having. It does nothing about a shell command, and a
        shell tool must therefore be gated by explicit human approval rather
        than by inspecting the command string.

        The layering to keep: **approval gates what runs; the sandbox bounds
        what running can reach.** They are different controls and neither
        substitutes for the other. A restricted PATH is a third thing —
        defence in depth at best, and dangerous if mistaken for either.
        """,
        "sandbox_is_kernel_policy",
    )


# ==========================================================================
# driver
# ==========================================================================

EXTRACTORS = {
    "concerns": extract_concerns,
    "crate_role": extract_crate_roles,
    "layering": extract_layering,
    "compare": extract_comparison,
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stage 2 — architecture analysis from grok-build-fork")
    ap.add_argument("--only", choices=sorted(EXTRACTORS))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--show", type=int, default=0)
    args = ap.parse_args()

    names = [args.only] if args.only else list(EXTRACTORS)
    total = 0
    for name in names:
        samples = list(EXTRACTORS[name]())
        if args.limit:
            samples = samples[: args.limit]
        out = STAGED / f"analysis_{name}.jsonl"
        n = write_jsonl(out, samples, secrets_on_hit="fail")
        total += n
        print(f"  {name:<11} {n:>4} samples -> {out.name}")

        for s in samples[: args.show]:
            d = s.to_dict()
            print("\n" + "=" * 72)
            print(f"id={d['id']}  meta={json.dumps(d['meta'], sort_keys=True)}")
            print(f"provenance={json.dumps(d['provenance'], sort_keys=True)}")
            for m in d["messages"]:
                if m["role"] == "system":
                    print(f"--- system --- [{len(m['content'])} chars, AUTHOR_SYSTEM]")
                    continue
                print(f"--- {m['role']} ---")
                print(m["content"])
            print("=" * 72)

    print(f"\ntotal: {total} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
