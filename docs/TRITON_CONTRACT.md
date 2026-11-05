# Triton Inference Contract

## Compatibility decision

The project pins `nvcr.io/nvidia/tritonserver:25.02-py3`, the final official
Triton server container that includes the TensorFlow backend. Newer Triton
containers cannot load `tensorflow_savedmodel` directly. Upgrading the server
therefore requires an explicit migration to ONNX, TensorRT, or a maintained
custom/Python backend; changing only the image tag is not safe.

## Versioned model repository

`make export-triton` reads the latest successful retrieval and ranking pointers
and creates this ignored binary layout:

```text
triton-models/
├── user_tower/config.pbtxt
├── user_tower/1/model.savedmodel/
├── wide_and_deep/config.pbtxt
├── wide_and_deep/1/model.savedmodel/
├── dcn/config.pbtxt
├── dcn/1/model.savedmodel/
└── manifest.json
```

The three configuration files and manifest are committed; version directories
are generated and ignored because they contain about 102 MiB of model binaries.
The manifest records source run IDs and SHA-256 checksums.

All models use version 1 inside a repository generated for one immutable pair
of retrieval/ranking run IDs. Model deployment identity still comes from the
manifest and API model version; the Triton directory number is an inference
server version, not the ML experiment identifier.

## Tensor contracts

- `user_tower`: `user_id` INT64 vector to `embedding` FP32 `[N, 64]`.
- `wide_and_deep` and `dcn`: twelve aligned ID/dense vectors to `logit` FP32
  `[N]`.
- `max_batch_size` is zero because the SavedModels already expose a dynamic
  leading dimension. Each request carries the complete candidate vector.
- CPU instances are requested so the internship demo does not require an
  NVIDIA GPU.

The API adapter uses Triton's V2 HTTP/JSON inference endpoint. It validates
model readiness, output presence, output shape, and finite values. Setting
`RECSYS_INFERENCE_BACKEND=triton` selects it; `local` remains available for
development and equivalence tests.

## Verification

`make verify-triton` sends identical user/ranker inputs to local SavedModel
signatures and live Triton, records maximum absolute error, and benchmarks 20
user-tower HTTP calls. A deployment is valid only when all three models are
ready and output differences remain within ordinary float32 tolerance.
