# WriteSAE

WriteSAE uses a sparse autoencoder to study what a recurrent language model stores in memory and how it affects predictions. This recurrent state is a matrix that carries information between tokens. We reconstruct saved states from a few active features, each shaped like a native memory write.

- Inspect which inputs activate a feature to study what information the state carries.
- Use the experiments to ablate or modify features in the state or its memory writes, then measure the effect on later predictions. Memory edits changed token scores as predicted (median R² = 0.98 on Qwen3.5-0.8B, layer 9, head 4). The language model weights stay frozen.

## Load a trained autoencoder

```bash
python -m pip install huggingface_hub torch
hf download JackYoung27/writesae-ckpts --local-dir writesae --include 'core/*' 'LOAD_EXAMPLE.py' 'writesae/qwen0p8b/L9_H4/*'
cd writesae
python LOAD_EXAMPLE.py
```

This example loads the Qwen3.5-0.8B layer 9, head 4 autoencoder on a CPU and checks the output shapes.

The memory-editing experiments are included in the Hugging Face release.

[MIT license](LICENSE).
