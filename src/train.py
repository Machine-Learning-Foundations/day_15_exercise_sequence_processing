"""Code to train generative LSTM-Language Models."""

import argparse
import pickle
from functools import partial
from typing import Any, Callable, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from flax.linen.initializers import orthogonal, zeros
from flax.linen.linear import Dense, default_kernel_init
from flax.linen.recurrent import RNNCellBase
from tqdm import tqdm

from utils import TextLoader

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any


def main():
    """Parse command line arguments and start the training."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/tinyshakespeare",
        help="data directory containing input.txt",
    )
    parser.add_argument(
        "--rnn_size", type=int, default=1024, help="size of RNN hidden state"
    )
    parser.add_argument("--model", type=str, default="lstm", help="rnn, gru, or lstm")
    parser.add_argument("--batch_size", type=int, default=100, help="minibatch size")
    parser.add_argument(
        "--seq_length", type=int, default=30, help="RNN sequence length"
    )
    parser.add_argument("--num_epochs", type=int, default=50, help="number of epochs")
    parser.add_argument(
        "--grad_clip", type=float, default=2.0, help="clip gradients at this value"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0015, help="learning rate"
    )
    args = parser.parse_args()
    train(args)


def convert(sequences: jnp.ndarray, inv_vocab: dict) -> list:
    """Convert an array of character-integers to a list of letters.

    Args:
        sequences (jnp.ndarray): An integer array, which represents characters.
        inv_vocab (dict): The dictonary with the integer to char mapping.

    Returns:
        list: A list of characters.
    """
    res = []
    # TODO: Write code to convert the network output back to a character sequence.
    return res


zero_key = jax.random.PRNGKey(0)


class LSTMCell(RNNCellBase):
    r"""LSTM cell.

    The mathematical definition of the cell is as follows

    .. math::
        \begin{array}{ll}
        i = \sigma(W_{ii} x + W_{hi} h + b_{hi}) \\
        f = \sigma(W_{if} x + W_{hf} h + b_{hf}) \\
        g = \tanh(W_{ig} x + W_{hg} h + b_{hg}) \\
        o = \sigma(W_{io} x + W_{ho} h + b_{ho}) \\
        c' = f * c + i * g \\
        h' = o * \tanh(c') \\
        \end{array}

    where x is the input, h is the output of the previous time step, and c is
    the memory.

    Attributes:
      gate_fn: activation function used for gates (default: sigmoid)
      activation_fn: activation function used for output and memory update
        (default: tanh).
      kernel_init: initializer function for the kernels that transform
        the input (default: lecun_normal).
      recurrent_kernel_init: initializer function for the kernels that transform
        the hidden state (default: orthogonal).
      bias_init: initializer for the bias parameters (default: zeros)
      dtype: the dtype of the computation (default: infer from inputs and params).
      param_dtype: the dtype passed to parameter initializers (default: float32).
    """
    gate_fn: Callable[..., Any] = nn.sigmoid
    activation_fn: Callable[..., Any] = nn.tanh
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    recurrent_kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = orthogonal()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, carry, inputs):
        r"""A long short-term memory (LSTM) cell.

        Args:
          carry: the hidden state of the LSTM cell,
            initialized using `LSTMCell.initialize_carry`.
          inputs: an ndarray with the input for the current time step.
            All dimensions except the final are considered batch dimensions.

        Returns:
          A tuple with the new carry and the output.
        """
        c, h = carry
        hidden_features = h.shape[-1]
        # input and recurrent layers are summed so only one needs a bias.
        dense_h = partial(
            Dense,
            features=hidden_features,
            use_bias=True,
            kernel_init=self.recurrent_kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        dense_i = partial(
            Dense,
            features=hidden_features,
            use_bias=False,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        # TODO: implement the LSTM equations here.
        new_c = c  # TODO: remove this line
        new_h = h  # TODOL remove this line
        return (new_c, new_h), new_h


class Model(nn.Module):
    """Creates an LSTM-based language model."""

    @nn.compact
    def __call__(self, carry, inputs, sample=False, key=zero_key) -> jnp.ndarray:
        """Create the network architecture in compact notation."""
        # Define your model using nn.LSTMcell and nn.Dense
        # inputs have shape [batch_size, time]

        cell = LSTMCell()
        # embed = nn.Dense(features=carry[0].shape[1])
        dense_out_projection = nn.Dense(features=65, use_bias=True)
        output_lst = []
        # output = embed(inputs[:, 0, :])
        output = inputs[:, 0, :]
        for t in range(inputs.shape[1]):
            if sample:
                cell_input = output
            else:
                cell_input = inputs[:, t, :]
                # emedded_input = embed(inputs[:, t, :])
                # current_key, key = jax.random.split(key)
                # rnd = jax.random.uniform(current_key, shape=[1])[0]
                # cell_input = jax.lax.cond(
                #    rnd > 0.3,
                #    lambda x: x[0],
                #    lambda x: x[1],
                #    (emedded_input,
                #     output) )
            carry, output = cell(carry, cell_input)
            projected_output = dense_out_projection(output)
            output = nn.one_hot(jnp.argmax(projected_output, -1), 65)
            output_lst.append(projected_output)

        return jnp.stack(output_lst, 1)


def train(args):
    """Train the model."""
    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length)

    key = jax.random.PRNGKey(42)  # type: ignore
    model = Model()

    net_state = model.init(
        key,
        (jnp.ones([1, args.rnn_size]), jnp.ones([1, args.rnn_size])),
        jnp.ones([1, args.seq_length, data_loader.vocab_size]),
    )
    lr = args.learning_rate
    opt = optax.adam(lr)
    opt_state = opt.init(net_state)

    @jax.jit
    def forward_step(net_state: FrozenDict, x: jnp.ndarray, y: jnp.ndarray, key):
        """Do a forward step."""
        out = model.apply(
            net_state,
            carry=carry,
            inputs=nn.one_hot(x, num_classes=data_loader.vocab_size),
            key=key,
        )
        ce_loss = jnp.mean(
            optax.softmax_cross_entropy(
                logits=out, labels=nn.one_hot(y, num_classes=data_loader.vocab_size)
            )
        )
        return ce_loss

    loss_grad_fn = jax.value_and_grad(forward_step)
    carry = (
        jnp.zeros([args.batch_size, args.rnn_size]),
        jnp.zeros([args.batch_size, args.rnn_size]),
    )

    def save():
        with open("./saved/net_state.pkl", "wb") as net_file:
            pickle.dump([model, net_state, args, data_loader], net_file)

    for e in range(args.num_epochs):
        data_loader.reset_batch_pointer()

        progress_bar = tqdm(
            range(data_loader.num_batches),
            desc="Training Language RNN",
        )
        for _ in progress_bar:
            x, y = data_loader.next_batch()
            current_key, key = jax.random.split(key)
            cel, grads = loss_grad_fn(net_state, x=x, y=y, key=current_key)
            grads = jax.tree_map(
                lambda x: jnp.clip(x, -args.grad_clip, args.grad_clip), grads
            )
            updates, opt_state = opt.update(grads, opt_state, net_state)
            net_state = optax.apply_updates(net_state, updates)
            progress_bar.set_description("Loss: {:2.3f}".format(cel))

        print("Epoch {} done".format(e))
        if e % 5 == 0:
            save()
            out = model.apply(
                net_state,
                carry=carry,
                inputs=nn.one_hot(x, num_classes=data_loader.vocab_size),
            )
            sequences = jnp.argmax(out, -1)
            seq_conv = convert(sequences, data_loader.inv_vocab)
            print("Poetry:")
            print("".join(seq_conv[0]))

        if e % 10 == 0:
            lr = lr / 1.1
            opt = optax.adam(lr)
            print("lr", lr)

    save()
    print("model saved.")


if __name__ == "__main__":
    main()
