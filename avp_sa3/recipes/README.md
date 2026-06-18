# Inference recipes

A catalog of **validated technique + parameter sets** for working with SA3/SAO at
inference — so the next agent (or the latent feature explorer) reuses what we tuned
instead of rediscovering it. The catalog is `inference_recipes.yaml`.

## What a recipe is

One named technique with its validated defaults, the tool that runs it, the venv, and
*when to reach for it*. Recipes are data, not code — the params are the source of truth
the tools and the explorer read.

```yaml
- id: flowsep-anchored          # stable key
  title: ...                    # human label
  kind: separation|riffer|steering|guidance|eval
  tool: avp_sa3/scripts/...     # entry point (path or model call)
  venv: sa3|mir|sat             # which interpreter (see top of the yaml)
  model: medium-base
  when: >                       # the one-line "use this for ..."
  params: {...}                 # the VALIDATED values
  notes: >                      # gotchas / what the dials do
  source: WORKLOG 2026-06-18    # where it was validated
```

## Current recipes
separation (`flowsep-anchored`, `zerosep-rf-eta`) · riffer (`riffer-audio-ref`,
`init-audio-style`) · steering (`chroma-steer`, `ridge-steer-explorer`) · guidance
(`latch-guided-sa3`) · eval (`stem-score`).

## Consumers

- **CLI tools** (`avp_sa3/scripts/*`, `sa3_control/generate.py`): use the `params` as
  starting defaults — the script flags mirror the recipe keys.
- **Latent feature explorer** (`mir/plots/explorer`, port 7895): the Viewer tab already
  applies *steering* recipes in-app (ridge directions, `z += strength·σ·β`). The
  integration path for the rest: a **recipe picker** in the Viewer that, for `kind:
  steering`, applies the direction live; for `kind: separation|riffer`, hands the
  selected track + recipe params to the SA3 `.venv` CLI and loads the result back into
  the player. The YAML is the shared contract between the two repos.

## Adding a recipe
Copy a block, set the validated params, cite the `source`. Keep the catalog current —
it's listed in the SAO-level `ARCHITECTURE.md` "reusable plumbing" index, so other
instances will look here before re-tuning.
