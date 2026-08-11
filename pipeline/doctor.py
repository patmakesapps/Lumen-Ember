"""Stage 0 self-check. Run this before any extraction stage.

Verifies the whole spine: sources cloned, licenses unchanged, Glimmer
reference files present and matching, LumaKit's registry importable, and the
secrets gate + sample validator actually catching what they claim to.

Run:  python -m pipeline.doctor
Exit: 0 = green, 1 = something is wrong (details printed)
"""

from __future__ import annotations

from pipeline import glimmer, provenance, registry_introspect, secrets_scan
from pipeline.config import GROK_REPO, LUMABOT_REPO, LUMAKIT_REPO, TRAINED_MODEL_NAME
from pipeline.sample import Sample, SampleInvalid, validate

OK = "  ok   "
BAD = "  FAIL "


def _check(results: list[tuple[bool, str]], cond: bool, msg: str) -> None:
    results.append((cond, msg))


def main() -> int:
    r: list[tuple[bool, str]] = []

    # --- sources -------------------------------------------------------
    _check(r, LUMAKIT_REPO.exists(), f"LumaKit checkout at {LUMAKIT_REPO}")
    _check(r, GROK_REPO.exists(), f"grok-build-fork checkout at {GROK_REPO}")
    _check(r, LUMABOT_REPO.exists(), f"LumaBot checkout at {LUMABOT_REPO}")

    try:
        provenance.verify_source_licenses()
        _check(r, True,
               "source licenses verified (LumaKit=MIT, fork=Apache-2.0, LumaBot=MIT)")
    except Exception as e:
        _check(r, False, f"license verification: {e}")

    # Not a hard failure — the owner authorised inclusion — but it must be
    # visible on every run rather than discovered at compliance-review time.
    unresolved = provenance.unresolved_licenses()
    if unresolved:
        print(f"  WARN  no LICENSE file in: {', '.join(unresolved)} — recorded as "
              f"'LicenseRef-Unspecified-OwnerAuthorized' and flagged in MIXTURE.md.\n"
              f"        Add a LICENSE (MIT would match LumaKit) to clear this.\n")

    lk_sha = provenance.git_sha(LUMAKIT_REPO)
    gb_sha = provenance.git_sha(GROK_REPO)
    _check(r, lk_sha != "unknown", f"LumaKit repo_sha = {lk_sha[:12]}")
    _check(r, gb_sha != "unknown", f"fork repo_sha    = {gb_sha[:12]}")

    # --- Glimmer reference --------------------------------------------
    try:
        info = glimmer.verify_reference_files()
        _check(
            r,
            not info["problems"],
            f"Glimmer ref: {info['model_type']} / {info['architecture']} "
            f"/ template {info['template_bytes']}B"
            + (f" — problems: {info['problems']}" if info["problems"] else ""),
        )
    except Exception as e:
        _check(r, False, f"Glimmer reference files: {e} (run pipeline.fetch_model_ref)")

    # --- LumaKit registry ---------------------------------------------
    try:
        payload = registry_introspect.dump_registry()
        n = len(payload["tools"])
        confirm = sum(1 for t in payload["tools"] if t["always_confirm"])
        _check(r, n > 50, f"LumaKit registry loads: {n} tools, {confirm} always-confirm")
    except Exception as e:
        _check(r, False, f"registry introspection: {e}")

    # --- secrets gate self-test ---------------------------------------
    # Test vectors are assembled at runtime so the literal patterns never
    # appear in this file. Otherwise the repo permanently trips its own
    # secrets gate — and GitHub push protection — on its own test fixtures.
    def _vec(*parts: str) -> str:
        return "".join(parts)

    positives = [
        {"messages": [{"role": "user", "content": _vec("AK", "IA", "IOSFODNN7EXAMPLE")}]},
        {"messages": [{"role": "user", "content": _vec("token = gh", "p_", "a" * 36)}]},
        {"messages": [{"role": "user",
                       "content": _vec("-----BEGIN ", "PRIVATE ", "KEY-----")}]},
        {"messages": [{"role": "user", "content": "api_key: " + "A1b2C3d4" * 4}]},
    ]
    caught = sum(1 for p in positives if secrets_scan.scan_sample(p))
    _check(r, caught == len(positives), f"secrets gate catches {caught}/{len(positives)} known keys")

    negatives = [
        {"messages": [{"role": "user", "content": "api_key: your_api_key_here"}]},
        {"messages": [{"role": "user", "content": "read_file on core/approval_policy.py"}]},
        {"messages": [{"role": "user", "content": "password = <redacted>"}]},
    ]
    fp = sum(1 for nn in negatives if secrets_scan.scan_sample(nn))
    _check(r, fp == 0, f"secrets gate false positives on placeholders: {fp}")

    # --- sample validator self-test ------------------------------------
    good = Sample(
        messages=[
            {"role": "user", "content": "read the readme"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": {"path": "README.md"}},
                }],
            },
            {"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": "# LumaKit"},
            {"role": "assistant", "content": "It's the LumaKit readme."},
        ],
        provenance=provenance.lumakit("doctor:self-test"),
    ).to_dict()
    try:
        validate(good)
        _check(r, True, "sample validator accepts a well-formed tool episode")
    except SampleInvalid as e:
        _check(r, False, f"validator rejected a good sample: {e}")

    # The one that matters: JSON-string arguments must be rejected, because
    # Glimmer's template raise_exception()s on them at train time.
    bad = dict(good)
    bad["messages"] = [dict(m) for m in good["messages"]]
    bad["messages"][1] = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "c1",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"path": "README.md"}'},
        }],
    }
    try:
        validate(bad)
        _check(r, False, "validator ACCEPTED JSON-string arguments (would crash training)")
    except SampleInvalid:
        _check(r, True, "sample validator rejects JSON-string tool arguments")

    # Glimmer does not support parallel tool calls, and its chat template does
    # not enforce that — it renders them happily. Must be caught here.
    par = dict(good)
    par["messages"] = [dict(m) for m in good["messages"]]
    par["messages"][1] = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "read_file", "arguments": {"path": "a.md"}}},
            {"id": "c2", "type": "function",
             "function": {"name": "read_file", "arguments": {"path": "b.md"}}},
        ],
    }
    try:
        validate(par)
        _check(r, False, "validator ACCEPTED parallel tool calls (unsupported by Glimmer)")
    except SampleInvalid:
        _check(r, True, "sample validator rejects parallel tool calls in one turn")

    # --- report --------------------------------------------------------
    print(f"Lumen-Ember pipeline doctor  (target: {TRAINED_MODEL_NAME})")
    print("-" * 68)
    for ok, msg in r:
        print((OK if ok else BAD) + msg)
    failures = [m for ok, m in r if not ok]
    print("-" * 68)
    if failures:
        print(f"{len(failures)} check(s) failed.")
        return 1
    print(f"All {len(r)} checks passed. Stage 0 is green.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
