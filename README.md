# BETDQNet: Bellman Error and TD trade-off Q-Network

## Overview

This repository contains the PyTorch implementation of **BETDQNet**, which leverages a combination of Bellman error (BE) and Temporal Difference (TD) error to prioritize experience samples during training. The two error types are dynamically weighted through a gradient-based optimization mechanism, with the aim of first encouraging exploration and then shifting towards exploitation.

## Key Features
- **Priority Sampling:** BETDQNet utilizes prioritized replay memory, giving higher priority to samples with larger TD and BE errors.
- **Dynamic Weight Adjustment:** The weights for TD and BE errors are dynamically adjusted during training.
- **Exploration and Exploitation:** The agent starts with a focus on exploration and gradually shifts toward exploitation as training progresses.
- **CartPole-v0 Environment:** The implementation is compatible with the CartPole-v0 environment from OpenAI Gym, though it can be extended to other environments.

## Installation

To run the code, ensure you have the following prerequisites installed:

1. Python 3.x
2. PyTorch
3. OpenAI Gym
4. NumPy

You can install the dependencies using the following command:

```bash
pip install torch gym numpy
```

## How it Works

### 1. Q-Network Architecture
The Q-network used in BETDQNet is a simple feedforward neural network for the OpenAI Gym environments and a CNN for the MinAtar experiments. It outputs Q-values for each action in the given state space.

### 2. BETDQNet Prioritization
Each sample added to the replay memory is accompanied by its weighted error score, combining the TD error and BE error, controlled by the weights `w1` and `w2`.

### 3. Training Process
The training process follows an epsilon-greedy exploration strategy.

### 4. Gradient-Based Weight Optimization
Weights assigned to TD error (`w1`) and BE error (`w2`) are adjusted through gradient-based optimization at each episode. 

## How to Use

### Training the Agent

To train the BETDQNet agent, simply run the provided script. The agent is configured to train on the CartPole-v0 environment.

```bash
python BETDQNet.py
```

The script runs for a total of 250 episodes by default, though this can be adjusted in the `EPISODES` variable.

## Citation
The codes provided in this repository support the research findings detailed in the following paper:

Hassani, Hossein, Soodeh Nikan, and Abdallah Shami. "Improved Exploration–Exploitation Trade-Off through Adaptive Prioritized Experience Replay." Neurocomputing 614 (2025): 128836.

Please cite this paper if you use this repository in your research.

## Acknowledgements

The PER memory used in this implementation is based on [rlcode/per](https://github.com/rlcode/per).

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
