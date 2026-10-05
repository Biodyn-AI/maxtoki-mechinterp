# Model checkpoints

The model weights are **not** in this release. Download them from the sources below and check
the sha256 values. All values come from `projects/maxtoki/checks/deployment_facts.json`
(section `checkpoints`), which hashed the local files and compared them with the Hugging Face API.

## MaxToki (used by every pipeline)

- Hugging Face repo: `theodoris-lab/MaxToki` (subfolders `MaxToki-217M-HF/` and `MaxToki-1B-HF/`).
- Revision to use: `21aa7b7f844c146c16fb79ffb3aa75d9b07a3d77` (2026-04-02, "add manuscript link").
- This is a **post-hoc pin**: no revision was written down at download time; the download used resolve/main. That commit was `main` when the files were downloaded (2026-04-16 00:00:58).
- Same bytes were also found at other revisions (check of 2026-10-01): `1523f7c2`: 217M True, 1B folder absent at this revision; `21aa7b7f`: 217M True, 1B True; `e7041eb7`: 217M True, 1B True; `main`: 217M True, 1B True.
- The runs used the Hugging Face safetensors files, not the BioNeMo checkpoint.

| File | Size (bytes) | sha256 | git blob sha1 | Matches HF at pin |
|---|---|---|---|---|
| `MaxToki-217M-HF/model.safetensors` | 867,797,760 | `9da1fbf8cc489486d158d4653a14e3f44ac49c75a8e1480566eab0e9f99cf898` | `b8f08f202e3f4c5ddf758bf7e36b965c6ed8f991` | True |
| `MaxToki-217M-HF/config.json` | 676 | `a4effbdf27969b7cc4d6cce34fc24b3f2134886946a19e7999c3ebd08babf9bd` | `d55a18cd382c0eff3c9fca48bd88df7c356e2f34` | True |
| `MaxToki-217M-HF/generation_config.json` | 132 | `3ecb636c810f6805b0534c7e8f6df2c53788726ec81b4b6372de0fed944d458e` | `f718cbb634ad6adf7cee66f5801a429128fef391` | True |
| `MaxToki-1B-HF/model.safetensors` | 4,196,167,128 | `96c6c8f9d61732c9bedfc37a9cfd937359e6ce0bfa860526646d101815a242e6` | `39ae124dfcd68b65d98bf814a76a9e3cd3212007` | True |
| `MaxToki-1B-HF/config.json` | 823 | `23358e6663b32e796dec2818a1d9cff97eb9e150d259850ba26b1bf2144eba9e` | `7b0e9401b2fb5c1a210fb1662b9ed3846308ab89` | True |
| `MaxToki-1B-HF/generation_config.json` | 111 | `b10a73d1785bb92ee3ff879dc114a6cb3adcf84caa3a4d31973b301cf059ddec` | `9142117462ac69b2ff300000fe68d619fa7674ab` | True |

Place the files at `projects/maxtoki/setup/MaxToki-217M-HF/` and `projects/maxtoki/setup/MaxToki-1B-HF/`.
Example (Python, `huggingface_hub`):

```python
from huggingface_hub import hf_hub_download
for sub in ("MaxToki-217M-HF", "MaxToki-1B-HF"):
    for f in ("config.json", "generation_config.json", "model.safetensors"):
        hf_hub_download("theodoris-lab/MaxToki", f"{sub}/{f}", revision="21aa7b7f844c146c16fb79ffb3aa75d9b07a3d77",
                        local_dir="projects/maxtoki/setup")
```

Then check: `shasum -a 256 projects/maxtoki/setup/MaxToki-*/model.safetensors`.

## Geneformer V2-316M (cross-model comparisons)

- Hugging Face repo: `ctheodoris/Geneformer`, subfolder `Geneformer-V2-316M/`.
- Revision: `05fcbeb8a27d49e0a7a4349152202ee2c1cbfd28` (HF lastModified 2026-02-06). This revision is written in the scripts as a cache snapshot path.
- Used by the deployment: `runs/spectral-geometry-217M/autoloop/iterations/iter_0003/h3del_lid_confound.py`, `runs/spectral-geometry-217M/scripts/audit_a3_bootstrap_cross_model_pearson.py`, `runs/spectral-geometry-217M/scripts/phase9b_cross_model.py`, `runs/topology-141-217M/scripts/phase14_cross_model_cca.py`.
- Used by later checks and the controlled studies: `runs/topology-141-217M/scripts/v2_crossmodel_common.py`, `v2_crossmodel_verify.py`, `runs/spectral-geometry-217M/scripts/v2_crossmodel_pearson_ci.py`, `verification/manifold_ci/04_crossmodel.py` (all under `projects/maxtoki/`), and `controlled-evaluation/studyA/build/T4/build_t4_packages.py` and `controlled-evaluation/studyA/keys/T4/verification/check_data.py`.

| File | Size (bytes) | sha256 |
|---|---|---|
| `Geneformer-V2-316M/model.safetensors` | 1,265,455,076 | `965ceccea81953d362081ef3843560a0e4fef88d396c28017881f1e94b1246f3` |
| `Geneformer-V2-316M/config.json` | 591 | `2cc4af3442644e84af71814a607b61c958d7b4e5ebde6791ab77b5f534ac6f6e` |

On 2026-10-01 the Hugging Face `main` branch served the same `model.safetensors` (compared by sha256). Use the pinned revision anyway, because `main` can change.

## scGPT whole-human (cross-model comparisons)

- scGPT is not on Hugging Face. The run used the official scGPT "whole-human" pretrained checkpoint
  (from the model list in the scGPT GitHub repository, bowang-lab/scGPT).
- `args.json` records `save_dir: /scratch/ssd004/datasets/cellxgene/save/cellxgene_census_human-May23-08-36-2023` (the upstream training path; kept as is).
- The deployment read only a gene-embedding table derived from the checkpoint, by `runs/topology-141-217M/scripts/phase4_scgpt_cross_model.py`.
- The later cross-model checks (`runs/topology-141-217M/scripts/v2_crossmodel_prepare.py`, `v2_crossmodel_common.py`, `v2_crossmodel_verify.py`) and the Study A T4 build (`controlled-evaluation/studyA/build/T4/build_t4_packages.py`) also read `best_model.pt`, `vocab.json` and `args.json`.

| File | Size (bytes) | sha256 |
|---|---|---|
| `best_model.pt` (whole-human checkpoint) | 205,385,258 | `6cb5d451ab5c4b33eb673adbe4fddc61d2389df1b89b7651a9fe2e557572b922` |
| `args.json` | 1,300 | `c18e075e018140cb8b2d9029387b9de26607a5ce6a8ccabd6ead70cd76b95d60` |
| `vocab.json` | 1,317,639 | `acca93d114ca62c3f0f50debbd23e8c87f0714f4737764454f6b2b13f2e8580f` |
| `scgpt_whole_human_gene_embeddings.pt` (derived table) | 249,944,756 | `b41289b862126ccd8db166e2688e08bcbe4149c87a9a177f8409b4723bf33f4b` |

The same `vocab.json` bytes are shipped as `controlled-evaluation/studyA/tasks/T4-*/data/scgpt_whole_human/vocab.json`.

The derived table lives at `<DATA_ROOT>/biodyn-work/subproject_53_scgpt_gpl_replication/embeddings/scgpt_whole_human_gene_embeddings.pt`. The script that made it
is outside this release (a separate project folder under `<DATA_ROOT>`).

## Tokenizer and gene-median files

| File | Shipped here? | Size (bytes) | sha256 |
|---|---|---|---|
| MaxToki token_dictionary.json | yes, `projects/maxtoki/setup/token_dictionary.json` | 560,316 | `3d06ba6f896c3c0f8721a78e39b68e660d9d12ea525af8c2ffdf631ac0a32d87` |
| Geneformer gene_median_dictionary_gc104M.pkl | no, `<DATA_ROOT>/biodyn-nmi-paper/src/02_cssi_method/crispri_validation/data/gene_median_dictionary_gc104M.pkl` | 1,512,661 | `a51c53f6a771d64508dfaf61529df70e394c53bd20856926117ae5d641a24bf5` |
| Geneformer gene_name_id_dict_gc104M.pkl | no, `<DATA_ROOT>/biodyn-nmi-paper/src/02_cssi_method/crispri_validation/data/gene_name_id_dict_gc104M.pkl` | 1,660,882 | `fabfa0c2f49c598c59ae432a32c3499a5908c033756c663b5e0cddf58deea8e1` |

The two gc104M `.pkl` files have the same sha256 as `geneformer/gene_median_dictionary_gc104M.pkl` and
`geneformer/gene_name_id_dict_gc104M.pkl` of `ctheodoris/Geneformer` at revision 05fcbeb8 (checked 2026-10-05).

The Geneformer gc104M median and name-to-id dictionaries come from the Geneformer repository
(`ctheodoris/Geneformer`). They were used in place of MaxToki's own gene medians, which are not published.
