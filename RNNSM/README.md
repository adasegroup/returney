# A Recurrent Neural Network Survival Model:Predicting Web User Return Time, Grob et al.

The authors noticed that the approach of Du et. al. does not consider users that did not return and that is why their predictions are biased in the left direction, meaning that they predict time of return that is on average lower than the actual one. They tried to tackle this problem in this paper.

### Problem statement

The problem that is solved in Grob et. al. is similar to that of Du et. al.: the input is a sequence of time/marker pairs, but the goal is to predict either the time of the next event OR that the user will not return during the prediction window.

### Mathematics

Mathematics of Grob et. al. is almost identical to that of Du et. al., so look it up in RMTPP folder. The only differences are the likelihood function and the prediction formula on inference. The log-likelihood function authors use here is ![](https://render.githubusercontent.com/render/math?math=\mathcal{l}(\mathcal{C})=\Sigma_{i}\Sigma_{j}\mathcal{l}(t_{j%2B1}^i),)
where ![](https://render.githubusercontent.com/render/math?math=\mathcal{l}(t_{j+1}^i)=\log[S^*(t_{j%2B1}^i)]) if i'th user did not return during the prediction window and j is this user's last recorded event and ![](https://render.githubusercontent.com/render/math?math=\mathcal{l}(t_{j+1}^i)=\log[f^*(t_{j%2B1}^i)]) otherwise. The functions f with a star and S with a star are exactly the same as in Du et. al.'s work.

On inference, they predict time between the last session and the next one as ![](https://render.githubusercontent.com/render/math?math=\widehat{d}_{j%2B1}=\mathbb{E}[\mathcal{T}\vert\mathcal{H}_{j}]=\int_{t_{j}}^{\infty}S^*(t)dt,) where the integral is computed numerically. However, this expression allows the model to predict return time before the start of prediction window, that is why they modify it as ![](https://render.githubusercontent.com/render/math?math=\mathbb{E}[\mathcal{T}|\mathcal{T}>t_{s}]=\frac{\int_{t_{s}}^{\infty}S^*(t)dt}{S(t_{s})}+\int_{0}^{t_{s}}S^*(t)dt.)