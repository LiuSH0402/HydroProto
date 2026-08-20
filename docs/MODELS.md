# Model definitions

All four models share an ESM-2 650M sequence encoder and the same five-class
hydrolase label space.

| Public name | Classification path | Local-positive objective | Online prototypes | Role |
|---|---|---:|---:|---|
| ESM2-MLP | MLP | No | No | Strong neural baseline |
| HydroLocal | MLP | Yes | No | Tests homology-aware local positive selection |
| HydroProto | Prototype scores | No | Yes | Main prototype-guided model |
| HydroMPC | MLP/prototype fusion | Yes | Yes | Combined ablation |

For HydroProto with `K=5`, the learnable prototype tensor has shape
`[5 classes, 5 prototypes per class, 128 dimensions]`, i.e. 25 class-conditional
prototypes. During inference, a sequence representation is compared with the
prototypes and class evidence is aggregated from the nearest prototypes. Training
uses classification, prototype-attraction and within-class diversity terms after a
prototype warm-up period.

The locked final configurations were selected exclusively by five-fold
cross-validation on the 2,721-record training set. They used 3, 6, 4 and 6 unfrozen
ESM-2 layers for ESM2-MLP, HydroLocal, HydroProto and HydroMPC, respectively.

Historical output folders may contain the old names `hlocal`, `hproto` and
`hydro_mpc`; these map to HydroLocal, HydroProto and HydroMPC and are retained only
to preserve provenance.
