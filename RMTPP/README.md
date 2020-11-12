# Du et al. Recurrent Marked Temporal Point Processes: Embedding Event History to Vector, Du et al.

## Problem
In the original paper the problem is the following: given a sequence of time/marker pairs, the goal is to predict the time/marer pair of the next event. Markers could be, for instance, actions a user took during the visit to the site.

The dataset is consists of time/marker pair sequences for a number of users.

## Mathematics

The likelihood function they are trying to maximize is $L(\theta) = \prod_{i}\prod_{j}f(t_{j+1}, y_{j+1}|\mathcal{H}_{j}) = \prod_{j}f^*(t_{j+1}, y_{j+1})$, where $f^*(t, y) = f(t, y|\mathcal{H}_{j}) = f(t, y|\mathbf{h}_{j})$ is the conditional density function that the next event with marker $y$ will happen at time $t$ given the compact representation $\mathbf{h}_{j}$ of history $\mathcal{H}_{j}$ up to time step $j$. The compact representation is obtained by processing the sequence with an RNN. They model this joint distribution as $f^*(t, y) = f^*(t)P(y_{j+1}=y|\mathbf{h}_{j})$.

To represent $f^*(t)$ using RNN's output, authors introduce $\lambda^*(t)$ function, which models the instantaneous rate of occurrence given that the event of interest did not occur until time $t$ conditioned on history. The relation between $f^*(t)$ and $\lambda^*(t)$ is the following: $\lambda^*(t) = \frac{f^*(t)}{S^*(t)}$, where $S^*(t)$ denotes the probability of the event of interest not having occurred by time $t$ conditioned on history. From the relation $\lambda^*(t) = -\frac{d\log S^*(t)}{dt}$, one can obtain $S^*(t) = \exp\big(-\int_{t_{j}}^{t}\lambda^*(t)\big)$, hence, $f^*(t) = \lambda^*(t)S^*(t) = \lambda^*(t)\exp\big(-\int_{t_{j}}^{t}\lambda^*(t)\big)$.

Finally, authors connect output of RNN to this notation as $\lambda^*(t) = \exp\big( \mathbf{v}^{(t)\top}\mathbf{h_{j}} + w(t-t_{j}) + b^{(t)}\big)$, where $h_{j}$ is RNN's hidden state after processing $H_{j}$, and $\mathbf{v}^{(t)}$, $w$ and $b^{(t)}$ are learnable parameters. 

During inference, they compute timing for the next event as $\widehat{t}_{j+1} = \int_{t_{j}}^{\infty}tf^*(t)dt$.

## Idea 

Main idea is to train a recurrent neural network to represent the given sequence as a vector and to predict both times and classes of actions using this vector embedding.

The architecture is the following:

![](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/fa58c1b3-ad8f-40cd-8723-38c17306f28c/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20201112%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20201112T114351Z&X-Amz-Expires=86400&X-Amz-Signature=81936c671466d571e6416ad8c2c660856e949e9e902288779b2e7ea5b77bfc50&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

1) Input is a sequence of pairs of time features (like time of event, time difference between current and previous event) and action features (e.g. one-hot encoding of action class)
2) Action features are passed through the learnable embedding layer, which allows to represent every action class in more meaningful way
3) Time and embedded action features are connected into one vector and passed into RNN
4) For every event in a series we extract the RNN embedding and predict time and action of the next event. This allows to learn on every event in the series (like in the language modeling problem)<br>
4.1. Action class is predicted using softmax classifier:<br>
![](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/be76559e-9dd9-475e-970f-fce4a30fe8bf/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20201112%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20201112T115340Z&X-Amz-Expires=86400&X-Amz-Signature=ce2582e9b2c34cfda975868659a6370b131ac1190f54ba30ed2527a3a42a5ae0&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)<br>
4.2. Time is predicted using conditional density function:
![](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/9e968164-2b99-4f49-920b-f507163a81bd/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20201112%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20201112T122341Z&X-Amz-Expires=86400&X-Amz-Signature=7baf7214a9f371c61d749dab265dbf3b50b9a5b7d660da828f12fa790d8586a0&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

Then, the network can be trained using BPTT, maximizing the following log-likelihood function:

![](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/f96e1c0a-564f-46bd-abce-b9b130d16f48/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20201112%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20201112T122554Z&X-Amz-Expires=86400&X-Amz-Signature=1503f131bb8fac146d3c5c6dfa6c74e7e75fd5e761ad38456e75411450e3c1f9&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

which is basically sum of loglosses for activity classification and for conditional density function of time.

