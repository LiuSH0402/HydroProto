# Configurations

The YAML files in this directory reproduce the frozen-embedding development stage
through `hydroproto embedding-train --config <file>` or
`hydroproto embedding-cv --config <file>`.
The ESM-2 650M partial fine-tuning commands use explicit CLI arguments so every
run writes its complete resolved configuration alongside the checkpoint.

Canonical public names are ESM2-MLP, HydroLocal, HydroProto and HydroMPC. Some
internal YAML `model` keys retain historical implementation identifiers so old
checkpoints remain loadable.
