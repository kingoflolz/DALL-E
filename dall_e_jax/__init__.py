import io, requests
import torch

import jax
import haiku as hk
import jax.numpy as jnp
from haiku._src.data_structures import FlatMapping
import numpy as np

from dall_e_jax.encoder import Encoder
from dall_e_jax.decoder import Decoder
from dall_e_jax.utils   import map_pixels, unmap_pixels


def load_statedict(path: str):
    if path.startswith('http://') or path.startswith('https://'):
        resp = requests.get(path)
        resp.raise_for_status()

        with io.BytesIO(resp.content) as buf:
            return torch.load(buf, map_location=torch.device("cpu")).state_dict()
    else:
        with open(path, 'rb') as f:
            return torch.load(f, map_location=torch.device("cpu")).state_dict()


def get_encoder(path: str):
    def encoder_jax(image):
        enc = Encoder(n_hid=256, n_blk_per_group=2, input_channels=3, vocab_size=8192)
        return enc(image)

    x = np.random.standard_normal((1, 3, 256, 256)).astype(np.float32)
    enc_transformed = hk.without_apply_rng(hk.transform(encoder_jax))
    jax_enc_params = enc_transformed.init(rng=jax.random.PRNGKey(0), image=x)
    jax_enc_params = convert_params(load_statedict(path), jax_enc_params)

    return enc_transformed.apply, jax_enc_params


def get_decoder(path: str):
    def decoder_jax(latent):
        dec = Decoder(n_init=128, n_hid=256, n_blk_per_group=2, output_channels=3, vocab_size=8192)
        return dec(latent)

    x = np.random.standard_normal((1, 8192, 32, 32)).astype(np.float32)
    dec_transformed = hk.without_apply_rng(hk.transform(decoder_jax))
    jax_dec_params = dec_transformed.init(rng=jax.random.PRNGKey(0), latent=x)
    jax_dec_params = convert_params(load_statedict(path), jax_dec_params)

    return dec_transformed.apply, jax_dec_params


def convert_params(torch_state, jax_params):
    def name_iter(pytree, root, f):
        new_out = {}
        for k, v in pytree.items():
            if isinstance(v, FlatMapping):
                new_out[k] = name_iter(v, root + "/" + k, f)
            else:
                new_out[k] = f(v, root + "/" + k)
        return new_out

    def process_node(value, name):
        name = name.lstrip("/")
        tensor_name = name.split("/")[-1]

        tensor_path = "/".join(name.split("/")[:-1])\
            .replace("/~/", ".")\
            .replace("/", ".")\
            .replace("encoder.", "")\
            .replace("decoder.", "")\
            .replace("~", "") \
            .replace("__", ".")

        pytorch_name = tensor_path + "." + tensor_name if tensor_path else tensor_name

        if tensor_name == "w":
            pytorch_tensor = torch_state[pytorch_name].permute([2, 3, 1, 0])
            new_tensor = jnp.array(pytorch_tensor)
        elif tensor_name == "b":
            pytorch_tensor = torch_state[pytorch_name].reshape(-1, 1, 1)
            new_tensor = jnp.array(pytorch_tensor)
        else:
            raise Exception("not implemented")

        assert new_tensor.shape == value.shape
        return new_tensor.astype("float32")

    return name_iter(jax_params, "", process_node)