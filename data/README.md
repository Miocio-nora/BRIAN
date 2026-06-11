# Data Directory

This directory is reserved for local data artifacts. Large generated data is ignored by Git.

Expected layout:

```text
data/
  manifests/    reproducibility manifests and stats
  raw/          downloaded or generated source text
  processed/    filtered text
  tokenized/    packed token files
  shards/       optional training shards
```

Every prepared recipe should produce a manifest and stats file under `data/tokenized/<recipe_name>/`.
