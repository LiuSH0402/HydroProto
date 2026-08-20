# Model weights

Large model weights are intentionally excluded from Git. Download
`HydroProto_models_v1.0.zip` from
[Zenodo (doi:10.5281/zenodo.22020952)](https://doi.org/10.5281/zenodo.22020952).

After extracting the archive, copy the four directories inside
`HydroProto_models_v1.0/models/` into this directory. The expected layout is:

```text
models/
|- README.md
|- esm2_mlp_unfreeze3_epoch11/
|  `- final_trainable_state.pt
|- hlocal_unfreeze6_epoch11/
|  `- final_trainable_state.pt
|- hproto_unfreeze4_epoch23/
|  `- final_trainable_state.pt
`- hydrompc_unfreeze6_epoch20/
   `- final_trainable_state.pt
```

Do not rename the model directories or remove their accompanying `config.json`,
metrics, predictions, audit, and history files.

| Public model | Directory | Weight SHA-256 |
|---|---|---|
| ESM2-MLP | `esm2_mlp_unfreeze3_epoch11/` | `c087eb5fee69e1c44e995008e9a0e3a6a2d868be4abe7d24cee49d145f1391eb` |
| HydroLocal | `hlocal_unfreeze6_epoch11/` | `82bd6a5f4e369aad49d45fce69e6f6a5cdf4ef1907bd68c6a7dd355bedaa8cfc` |
| HydroProto | `hproto_unfreeze4_epoch23/` | `6072842f7eb8dfdaeb7bfe1d1dac7f6dbfd658a201d30144531702eff05f003a` |
| HydroMPC | `hydrompc_unfreeze6_epoch20/` | `bb4fc1e1c36304899db9ad43e10c0ccc6a8c29e90fe3714baea85e7a44903163` |

The checkpoints contain the fine-tuned trainable layers and task heads but not the
complete ESM-2 backbone. Obtain `facebook/esm2_t33_650M_UR50D` separately, place it
at `../hf_cache/esm2_t33_650M_UR50D/`, and provide that path through `--model-dir`.

Historical directory names use `hlocal`, `hproto`, and `hydrompc`; their public
model names are HydroLocal, HydroProto, and HydroMPC. The four checkpoint files
have model-specific state-dictionary layouts, so load each one with its
corresponding model implementation.

Compact result tables and the archive description are available in
[`../results/RESULT_ARCHIVE.md`](../results/RESULT_ARCHIVE.md).
