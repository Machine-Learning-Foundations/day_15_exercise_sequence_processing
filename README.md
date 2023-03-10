# Exercise Generative Language Modelling

This exercise explores the use of LSTM-networks for the task of poetry generation.
To make it work finish `src/train.py`. These cells are typically defined as,

$$        \mathbf{z}_t = \tanh( \mathbf{W}_z \mathbf{x}_t + \mathbf{R}_z \mathbf{h}_{t-1}  + \mathbf{b}_z),  $$

$$\mathbf{i}_t =  \sigma( \mathbf{W}_i \mathbf{x}_t + \mathbf{R}_i \mathbf{h}_{t-1} + \mathbf{p}_i \odot \mathbf{c}_{t-1}+ \mathbf{b}_i), $$

$$  \mathbf{f}_t = \sigma(\mathbf{W}_f \mathbf{x}_t + \mathbf{R}_f \mathbf{h}_{t-1} + \mathbf{p}_f \odot \mathbf{c}_{t-1}+ \mathbf{b}_f), $$

$$ \mathbf{c}_t = \mathbf{z}_t \odot \mathbf{i}_t + \mathbf{c}_{t-1} \odot \mathbf{f}_t, $$

$$  \mathbf{o}_t = \sigma(\mathbf{W}_o \mathbf{x}_t + \mathbf{R}_o \mathbf{h}_{t-1} + \mathbf{p}_o \odot \mathbf{c}_t+ \mathbf{b}_o), $$

$$ \mathbf{h}_t = \tanh(\mathbf{c}_t) \odot \mathbf{o}_t. $$

The input as denotes as $\mathbf{x}_t \in \mathbb{R}^{n_i}$ it changes according to time $t$.
Potential new states $\mathbf{z}_t$ are called block input. 
$\mathbf{i}$ is called the input gate. The forget gate is $\mathbf{f}$ and
$\mathbf{o}$ denotes the output gate.
$\mathbf{p} \in \mathbb{R}^{n_h}$ are peephole weights,
$\mathbf{W} \in \mathbb{R}^{n_i \times n_h}$ denotes input,
$\mathbf{R} \in \mathbb{R}^{n_o \times n_h}$ are the recurrent matrices.
$\odot$ indicates element-wise products. 


When you have trained a model run `src/recurrent_poetry.py`, enjoy!

(Exercise inspired by: https://karpathy.github.io/2015/05/21/rnn-effectiveness/ )
