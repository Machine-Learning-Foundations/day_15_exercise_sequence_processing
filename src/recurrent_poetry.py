"""This module ships a function."""
import pickle

import jax
import jax.numpy as jnp

from train import Model, convert  # noqa: F401


def sample_stored_model(model_path):
    """Sample from a stored model."""
    with open(model_path, "rb") as weight_file:
        model, net_state, args, data_loader = pickle.load(weight_file)

    print("args", args)
    seq_len = 250
    # init_chars = jax.random.randint(key, shape=[4,], minval=0, maxval=65)
    init_list = [0, 1, 3, 9, 13, 24, 29, 42, 43, 44, 49, 58]
    # init_list = [0, 1, 13]
    init_chars = jnp.array(init_list)
    init_chars = jax.nn.one_hot(init_chars, num_classes=65)
    # print(init_chars.shape)
    init_chars = jnp.stack([init_chars] * seq_len, axis=1)
    # print(init_chars.shape)
    carry = (
        jnp.zeros([len(init_list), args.rnn_size]),
        jnp.zeros([len(init_list), args.rnn_size]),
    )

    sequences = model.apply(net_state, carry, init_chars, sample=True)
    sequences = jnp.argmax(sequences, -1)

    print(data_loader.inv_vocab)

    seq_conv = convert(sequences, data_loader.inv_vocab)
    for i in range(len(init_list)):
        input = data_loader.inv_vocab[init_list[i]]
        print('--> Input "{}" leads to poetry:'.format(input))
        print(input + "".join(seq_conv[i]))
        print("")


if __name__ == "__main__":
    model_path = "./saved/net_state.pkl"
    sample_stored_model(model_path)
