# These files are a handover, not part of the repo's test suite. The agent copies provided/ and provided_round2/
# into scripts/synth_inversion/ before using them (see SPEC.md). This keeps a repo-wide `pytest` run from collecting
# them by accident, where their imports (inharmonicity_target, surge_spec_v3, ...) do not resolve.
collect_ignore_glob = ["provided/*", "provided_round2/*"]
