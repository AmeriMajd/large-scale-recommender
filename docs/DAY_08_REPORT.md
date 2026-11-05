# Day 8 Report: Triton Inference Integration

## Outcome

Day 8 separates neural inference from recommendation orchestration. The API can
now call either the existing in-process TensorFlow SavedModels or a Triton V2
HTTP endpoint through one interface. An exporter builds a versioned Triton
repository for the user tower, Wide & Deep ranker, and DCN ranker, while a
verification command compares remote outputs with the local SavedModel
signatures and records HTTP latency.

The repository export and client contract are implemented and unit tested. The
official `nvcr.io/nvidia/tritonserver:25.02-py3` image is 10.8 GB compressed;
its pull was cancelled, so the live server equivalence test remains pending.
This distinction matters: mocked protocol tests prove request construction and
validation, but they do not prove that the server loads these SavedModels.

## Concepts learned

Model serving is a contract boundary. The caller and server must agree on model
names, versions, input names, data types, shapes, and output names. A valid
SavedModel is necessary but not sufficient; its exported signature must also
match `config.pbtxt` exactly.

Triton distinguishes request batching from a tensor's own leading dimension.
These SavedModels already accept dynamic candidate vectors, so
`max_batch_size: 0` tells Triton not to add another batch dimension. The API
sends one user ID to the tower and aligned vectors of candidates to a ranker.

Deployment identity is separate from the Triton numeric version. The generated
manifest records the retrieval and ranking run IDs plus SHA-256 hashes, while
the repository uses version `1` as the deployable snapshot for that manifest.

## Key code walkthrough

[`inference.py`](../src/recommender/serving/inference.py) defines the common
backend interface. `LocalTensorFlowInference` preserves the development path;
`TritonHttpInference` serializes NumPy arrays to Triton's V2 JSON protocol and
rejects unknown rankers, missing outputs, wrong shapes, and non-finite values.

[`engine.py`](../src/recommender/serving/engine.py) now owns retrieval and
recommendation logic without directly importing TensorFlow. Environment
variables select local or remote inference, and Triton selection fails closed
at startup if its health endpoint is not ready.

[`triton_repository.py`](../src/recommender/serving/triton_repository.py) reads
the latest immutable training pointers, copies all three SavedModels into
versioned directories, and writes an auditable manifest. Generated model
binaries are ignored; the small model configurations and manifest are kept in
Git.

[`verify_triton.py`](../src/recommender/serving/verify_triton.py) is the live
acceptance check. It sends identical examples through local TensorFlow and
Triton, checks maximum absolute error for all models, and measures user-tower
HTTP latency.

## Complete debugging log

| Symptom | Root cause | Fix | Lesson |
|---|---|---|---|
| Importing the API loaded TensorFlow even when remote inference was selected | TensorFlow was imported at module scope | Moved TensorFlow imports into the local adapter | Optional backends should be lazy dependencies |
| Ranking feature construction was tied to TensorFlow tensors | The feature store created tensors before choosing an inference backend | Made the feature boundary NumPy-based and let each adapter serialize it | Keep shared feature contracts framework-neutral |
| The latest Triton releases no longer contain the TensorFlow backend | NVIDIA removed that backend from newer server containers | Pinned the final official TensorFlow-capable release, 25.02 | Compatibility is a tested deployment decision, not just a newest-version choice |
| Two ranker configuration files contained a literal leading `+` | A patch marker was accidentally saved as content | Removed it before commit and revalidated the repository configuration | Inspect generated/deployment text as well as Python tests |
| The official Triton pull took too long | The compressed image is 10.8 GB and contains CUDA plus multiple inference backends | Cancelled the pull and kept the live test service-gated | Do not confuse implemented integration code with completed live validation |

## Tests and evidence

- Unit tests cover Triton request data types, shapes, outputs, health checks,
  unknown variants, and versioned repository export.
- `make export-triton` generated all three model directories and a manifest for
  the latest retrieval/ranking run IDs.
- The local recommendation path remains covered by the full repository suite.
- The live equivalence test is intentionally skipped unless
  `RUN_TRITON_INTEGRATION=1` and a ready server are present.

Not yet proven: live model loading, local-versus-Triton numerical equivalence,
Triton HTTP latency, concurrency, dynamic batching, GPU execution, or server
failure recovery.

## Learning resources, in order

1. [Triton model repositories](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html)
   — understand model names, numeric versions, and repository layout.
2. [Triton model configuration](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html)
   — focus on tensor contracts and batching rules.
3. [Deploying a TensorFlow model](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/Quick_Deploy/TensorFlow/README.html)
   — connect SavedModel signatures to server configuration.
4. [Triton release notes](https://docs.nvidia.com/deeplearning/triton-inference-server/pdf/Triton-Inference-Server-Release-Notes.pdf)
   — review backend compatibility before changing the pinned image.

## Practice exercises

1. Explain why `max_batch_size: 0` is correct for the exported signatures.
2. What should happen if one ranker is ready but the other one is not?
3. Add an HTTP timeout test and decide whether ranking should retry.
4. Compare model version `1` with the retrieval/ranking run IDs in the manifest.
5. When the image is available, run the live test and explain any numerical
   difference larger than `1e-5`.

## Reproduction

```bash
make export-triton
docker compose up -d triton
make test-triton
make verify-triton
```

The last three commands require the 10.8 GB Triton image. See
[`TRITON_CONTRACT.md`](TRITON_CONTRACT.md) for the complete serving contract.
