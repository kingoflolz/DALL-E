import io

import jax
import requests
import PIL
from PIL import ImageOps

import numpy as np
import jax.numpy as jnp

from dall_e_jax import get_encoder, get_decoder, map_pixels, unmap_pixels

target_image_size = 256


def download_image(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return PIL.Image.open(io.BytesIO(resp.content))


def preprocess(img):
    img = ImageOps.fit(img, [target_image_size,] * 2, method=0, bleed=0.0, centering=(0.5, 0.5))

    img = np.expand_dims(np.transpose(np.array(img).astype(np.float32)/255, (2, 0, 1)), 0)
    return map_pixels(img)


jax_enc_fn, jax_enc_params = get_encoder("encoder.pkl")
jax_dec_fn, jax_dec_params = get_decoder("decoder.pkl")

x = preprocess(download_image('https://assets.bwbx.io/images/users/iqjWHBFdfxIU/iKIWgaiJUtss/v2/1000x-1.jpg'))

z_logits = jax_enc_fn(jax_enc_params, x)

z = jnp.argmax(z_logits, axis=1)
z = jnp.transpose(jax.nn.one_hot(z, num_classes=8192), (0, 3, 1, 2))

x_stats = jax_dec_fn(jax_dec_params, z)

x_rec = unmap_pixels(jax.nn.sigmoid(x_stats[:, :3]))
x_rec = np.transpose((np.array(x_rec[0]) * 255).astype(np.uint8), (1, 2, 0))

PIL.Image.fromarray(x_rec).save('reconstructed.png')
