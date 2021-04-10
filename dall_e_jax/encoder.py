from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp


class EncoderBlock(hk.Module):
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
                hk.Conv2D(n_hid, 3, name="conv_1", data_format="NCHW"),
                jax.nn.relu,
                hk.Conv2D(n_hid, 3, name="conv_2", data_format="NCHW"),
                jax.nn.relu,
                hk.Conv2D(n_hid, 3, name="conv_3", data_format="NCHW"),
                jax.nn.relu,
                hk.Conv2D(n_out, 1, name="conv_4", data_format="NCHW")])

    def __call__(
            self,
            x: jnp.ndarray,
    ) -> jnp.ndarray:
        return self.id_path(x) + self.post_gain * self.res_path(x)


class Encoder(hk.Module):
    def __init__(
            self,
            n_hid: int,
            n_blk_per_group: int,
            input_channels: int,
            vocab_size: int,
    ):
        super().__init__()
        self.input_channels = input_channels
        group_count = 4

        blk_range = range(n_blk_per_group)
        n_layers = group_count * n_blk_per_group
        make_blk = partial(EncoderBlock, n_layers=n_layers)
        maxpool = partial(hk.max_pool, window_shape=2, strides=2, channel_axis=1, padding="SAME")

        with hk.experimental.name_scope("blocks"):
            self.blocks = hk.Sequential([
                hk.Conv2D(n_hid, 7, name="input", data_format="NCHW"),
                hk.Sequential([
                    *[make_blk(1 * n_hid, 1 * n_hid, name=f'group_1__block_{i + 1}') for i in blk_range],
                    maxpool
                ]),
                hk.Sequential([
                    *[make_blk(1 * n_hid if i == 0 else 2 * n_hid, 2 * n_hid, name=f'group_2__block_{i + 1}') for i in blk_range],
                    maxpool
                ]),
                hk.Sequential([
                    *[make_blk(2 * n_hid if i == 0 else 4 * n_hid, 4 * n_hid, name=f'group_3__block_{i + 1}') for i in blk_range],
                    maxpool
                ]),
                hk.Sequential([
                    *[make_blk(4 * n_hid if i == 0 else 8 * n_hid, 8 * n_hid, name=f'group_4__block_{i + 1}') for i in blk_range],
                ]),
                hk.Sequential([
                    jax.nn.relu,
                    hk.Conv2D(vocab_size, 1, name="output__conv", data_format="NCHW"),
                ]),
            ])

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if len(x.shape) != 4:
            raise ValueError(f'input shape {x.shape} is not 4d')
        if x.shape[1] != self.input_channels:
            raise ValueError(f'input has {x.shape[1]} channels but model built for {self.input_channels}')
        if x.dtype != jnp.float32:
            raise ValueError('input must have dtype torch.float32')

        return self.blocks(x)
