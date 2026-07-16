# BlockMGARD

Experiments for the BlockMGARD paper.

## Reproduce

zfp baseline (edit the paths at the top of the script for your machine):

```bash
./scripts/zfp_comp_repro.sh                     # run everything
./scripts/zfp_comp_repro.sh NYX Miranda         # only these datasets
ERR=1e-4 ./scripts/zfp_comp_repro.sh            # only one error level
DRY_RUN=1 ./scripts/zfp_comp_repro.sh           # print commands, don't run
```

Datasets (SDRBENCH) and the zfp binary are not tracked in this repo; set
`SDR_ROOT` and `zfp_exec` inside the script to point at your local copies.
