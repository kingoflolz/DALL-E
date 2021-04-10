from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp


class DecoderBlock(hk.Module):
    def __init__(
            self,
            n_in: int,
            n_out: int,
            n_layers: int,
            name: str = "EncoderBlock"
    ):
        super().__init__(name=name)
        n_hid = n_out // 4
        self.post_gain = 1 / (n_layers ** 2)

        self.id_path = hk.Conv2D(n_out, 1, name="id_path", data_format="NCHW") if n_in != n_out else lambda x: x

        with hk.experimental.name_scope("res_path"):
            self.res_path = hk.Sequential([
                jax.nn.relu,
                hk.Conv2D(n_hid, 1, name="conv_1", data_format="NCHW"),
                jax.nn.relu,
                hk.Conv2D(n_hid, 3, name="conv_2", data_format="NCHW"),
                jax.nn.relu,
                hk.Conv2D(n_hid, 3, name="conv_3", data_format="NCHW"),
                jax.nn.relu,
                hk.Conv2D(n_out, 3, name="conv_4", data_format="NCHW")])

    def __call__(
            self,
            x: jnp.ndarray,
    ) -> jnp.ndarray:
        return self.id_path(x) + self.post_gain * self.res_path(x)


class Decoder(hk.Module):
    def __init__(
            self,
            n_init: int,
            n_hid: int,
            n_blk_per_group: int,
            output_channels: int,
            vocab_size: int
    ):
        super().__init__()
        group_count = 4
        self.vocab_size = vocab_size

        blk_range = range(n_blk_per_group)
        n_layers = group_count * n_blk_per_group
        make_blk = partial(DecoderBlock, n_layers=n_layers)

        def upsample(x):
            s = x.shape
            return jax.image.resize(x, shape=(s[0], s[1], s[2] * 2, s[3] * 2), method="nearest")

        with hk.experimental.name_scope("blocks"):
            self.blocks = hk.Sequential([
                hk.Conv2D(n_init, 1, name="input", data_format="NCHW"),
                hk.Sequential([
                    *[make_blk(n_init if i == 0 else 8 * n_hid, 8 * n_hid, name=f'group_1__block_{i + 1}') for i in
                      blk_range],
                    upsample
                ]),
                hk.Sequential([
                    *[make_blk(8 * n_hid if i == 0 else 4 * n_hid, 4 * n_hid, name=f'group_2__block_{i + 1}') for i in
                      blk_range],
                    upsample
                ]),
                hk.Sequential([
                    *[make_blk(4 * n_hid if i == 0 else 2 * n_hid, 2 * n_hid, name=f'group_3__block_{i + 1}') for i in
                      blk_range],
                    upsample
                ]),
                hk.Sequential([
                    *[make_blk(2 * n_hid if i == 0 else 1 * n_hid, 1 * n_hid, name=f'group_4__block_{i + 1}') for i in
                      blk_range],
                ]),
                hk.Sequential([
                    jax.nn.relu,
                    hk.Conv2D(2 * output_channels, 1, name="output__conv", data_format="NCHW"),
                ]),
            ])

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if len(x.shape) != 4:
            raise ValueError(f'input shape {x.shape} is not 4d')
        if x.shape[1] != self.vocab_size:
            raise ValueError(f'input has {x.shape[1]} channels but model built for {self.vocab_size}')
        if x.dtype != jnp.float32:
            raise ValueError('input must have dtype torch.float32')

        return self.blocks(x)
