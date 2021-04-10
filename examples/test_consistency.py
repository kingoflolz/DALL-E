import io
import requests
import PIL

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F

from dall_e import map_pixels, load_model
from dall_e_jax import get_encoder, get_decoder

target_image_size = 256


def download_image(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return PIL.Image.open(io.BytesIO(resp.content))


def preprocess(img):
    s = min(img.size)

    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')

    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    return map_pixels(img)


pytorch_enc = load_model("encoder.pkl", torch.device('cpu'))
pytorch_dec = load_model("decoder.pkl", torch.device('cpu'))

jax_enc_fn, jax_enc_params = get_encoder("encoder.pkl")
jax_dec_fn, jax_dec_params = get_decoder("decoder.pkl")

x = preprocess(download_image('https://assets.bwbx.io/images/users/iqjWHBFdfxIU/iKIWgaiJUtss/v2/1000x-1.jpg'))

z_logits_pytorch = pytorch_enc(x)
z_logits_jax = jax_enc_fn(jax_enc_params, x.detach().numpy())

assert np.allclose(z_logits_jax, z_logits_pytorch.detach().numpy(), atol=1e-2, rtol=1e-2)

z = torch.argmax(z_logits_pytorch, axis=1)
z = F.one_hot(z, num_classes=pytorch_enc.vocab_size).permute(0, 3, 1, 2).float()

x_stats_pytorch = pytorch_dec(z).float()
x_stats_jax = jax_dec_fn(jax_dec_params, z.float().detach().numpy())

assert np.allclose(x_stats_jax, x_stats_pytorch.detach().numpy(), atol=1e-2, rtol=1e-2)

print("check ok!")
