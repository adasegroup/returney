# Du et al. Recurrent Marked Temporal Point Processes: Embedding Event History to Vector, Du et al.

### Problem statement
The problem stated in the paper is as follows: given a sequence of events, each of which is characterized by a set of markers and the time at which it happened, the goal is to predict the time of the next event. Markers could be, for instance, actions a user took during the visit to the site.
The dataset consists of time/marker pair sequences for a number of users.

### Mathematics
The likelihood function they are trying to maximize is
![](https://render.githubusercontent.com/render/math?math=L(\theta)=\prod_{i}\prod_{j}f(t_{j%2B1},y_{j%2B1}\vert\mathcal{H}_{j})=\prod_{j}f(t_{j%2B1},y_{j%2B1}),f^*(t,y)=f(t,y\vert\mathcal{H}_{j})=f(t,y\vert\mathbf{h}_{j}),)
where the f with a star is the conditional density function that the next event with marker y will happen at time t given the compact representation h of history H up to time step j. The compact representation is obtained by processing the sequence with an RNN. They model this joint distribution as ![](https://render.githubusercontent.com/render/math?math=f^*(t,y)=f^*(t)P(y_{j%2B1}=y\vert\mathbf{h}_{j}).) One can notice that markers are also included in the likelihood: indeed, they train their RNN on the task of prediction of markers of future events, which serves an auxiliary task to the main one.

To represent f with a star using RNN's output, authors introduce the following function: ![](https://render.githubusercontent.com/render/math?math=\lambda^*(t)=\frac{f^*(t)}{S^*(t)},) where S with a star denotes the probability of the event of interest not having occurred by time t conditioned on history. The lambda with a star models the instantaneous rate of occurrence given that the event of interest did not occur until time t, also conditioned on history. From the relation ![](https://render.githubusercontent.com/render/math?math=\lambda^*(t)=-\frac{d\logS^*(t)}{dt},) one can obtain ![](https://render.githubusercontent.com/render/math?math=S^*(t)=\exp[-\int_{t_{j}}^{t}\lambda^*(t)dt],)
hence, ![](https://render.githubusercontent.com/render/math?math=f^*(t)=\lambda^*(t)S^*(t)=\lambda^*(t)\exp[-\int_{t_{j}}^{t}\lambda^*(t)].)

Finally, authors connect output of RNN to this notation as ![](https://render.githubusercontent.com/render/math?math=\lambda^*(t)=\exp[\mathbf{v}^{(t)\top}\mathbf{h_{j}}%2Bw(t-t_{j})%2Bb^{(t)}]), where h is RNN's hidden state after processing history H, and v, w, and b are learnable parameters. 

During inference, they compute timing for the next event as ![](https://render.githubusercontent.com/render/math?math=\widehat{t}_{j%2B1}=\int_{t_{j}}^{\infty}tf^*(t)dt,) which could be approximated using numerical integration techniques.

### Idea 

As we have said, the main idea of the approach is to train an RNN that can assign a meaningful representation to a sequence (i.e. a vector), which can then be used to predict the timing of the next event.

The architecture of their RNN looks as follows:

![](https://sun9-37.userapi.com/impf/i_He6l3FVAHwYAV_1A-BUHLmd6iXXC5H9Ztwuw/2osvj6iTEHI.jpg?size=611x377&quality=96&proxy=1&sign=d4c3a0900322f0928c2db1ee31ba90a6)

1. Input is a sequence of pairs of time features (e.g. time of the event, time difference between the current and the previous event) and marker features (e.g. one-hot encoding of a class of an action taken at that time)  
2. Categorical marker features are passed through the learnable embedding layer
3. Time and embedded marker features are connected into one vector and passed as input to the RNN
4. At every time step, after processing new data, the RNN updates its current hidden representation. This hidden representation is then passed to two fully connected layers: the first one is used to compute logits for the prediction of the class of the marker that is associated to the next event, and the second one is used to predict the timing of the next event. With this kind of modeling, the RNN can learn on every event in the series (like in the language modeling problem)
4.1. Marker class is predicted using softmax classifier:
![](https://sun9-4.userapi.com/impf/WmpweOvUtyKD1ArYEFfZXaVX9ppcCQ7kAtQFAw/lboxCUeuzks.jpg?size=451x106&quality=96&proxy=1&sign=2a506f05c2cf33aa07e3919d7fd52d8c)
4.2. Time is predicted using conditional density function:
![](https://sun9-53.userapi.com/impf/FPp5GrR5LXop5YQNw-D_PNTrRNCFxxskGzLA_Q/vhp6w1fGVEw.jpg?size=625x230&quality=96&proxy=1&sign=65349cb9a4333b0ead18d3f1195d2c95)

The RNN is trained using BPTT, maximizing the following log-likelihood function:

![](https://sun9-56.userapi.com/impf/sXswa1ilkzoQEv2HveTBI912OnK7UWUWBZKv3g/QY7gzFfPdM4.jpg?size=535x68&quality=96&proxy=1&sign=f34904381dc8e2cbacf27c019680a804)

The d here stands for the time difference between (j+1)-th and j-th events.
 

