"""Lumen-Ember dataset pipeline.

Builds a QLoRA fine-tuning dataset that turns Meta's Muse Glimmer 30B into
Lumen-Ember-30B: a LumaKit-native local agent model.

Stages (each independently runnable and idempotent):
    0. pipeline.config / provenance / secrets_scan / sample  — shared spine
    1. pipeline.extract_static      — static extraction from LumaKit
    2. pipeline.extract_analysis    — architecture analysis from grok-build-fork
    3. pipeline.gen_trajectories    — synthetic agent episodes (core asset)
    4. pipeline.filter              — schema validation, judge, dedup, decontam
    5. pipeline.mix_external        — external tool-calling corpora + final mix
    6. pipeline.eval                — base vs LoRA eval harness

Every stage writes JSONL where each line is a Sample (see pipeline.sample)
carrying provenance metadata. No sample may leave a stage without it.
"""

__version__ = "0.1.0"
