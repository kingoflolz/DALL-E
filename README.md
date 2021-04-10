# Overview

This is an unofficial port of the dalle encoder/decoder models and checkpoints to Jax + Haiku, while preserving the same output. See `examples/test_consistency.py` for details.

To run the examples, the checkpoints must be downloaded into the examples folder:
```shell
cd examples
wget https://cdn.openai.com/dall-e/encoder.pkl
wget https://cdn.openai.com/dall-e/decoder.pkl
```

# TODO
- [ ] Test/benchmark on TPUs
- [ ] Add dataloading code to bulk preprocess tfrecord datasets for dall-e training
