# AIED2026: 动态 SCM × 序列生成 × 因果约束（PyTorch 代码骨架）

本仓库提供一个可扩展的最小实现骨架，用于把教育场景的动态 SCM（潜在知识状态 `K_t`）嵌入序列生成训练框架，并暴露 `do(T_t=a)` / `policy(h_t)` 可干预接口，同时加入可写进论文的因果约束损失。

## 目录结构

```
src/
  data.py
  model/
    scm_generator.py
    policy.py
    discriminators.py
    losses.py
  train.py
  eval/
    interventional_eval.py
    policy_sim.py
```

## 数据格式（.npz，最小可行）

`np.savez("data.npz", X=..., A=..., T=..., Y=..., M=...)`

- `X`: `[N, d_x]` 静态协变量
- `A`: `[N, T]`（离散 id）或 `[N, T, d_a]`（连续特征）
- `T`: `[N, T]`（离散动作）或 `[N, T, d_t]`（连续动作）
- `Y`: `[N, T]` 或 `[N, T, d_y]`（当前默认按二元 `Bernoulli` 训练）
- `M`: `[N, T]` mask（可选；缺省为全 1）

## 训练（Stage A/B/C）

安装依赖：`pip install -r requirements.txt`

使用合成数据：

- IRT-like（更贴近 AIED；训练脚本会默认开启 `--y_in_dynamics`）：`python -m src.train --out_dir runs/synth1 --synthetic_kind irt`
- Logistic baseline：`python -m src.train --out_dir runs/synth0 --synthetic_kind logistic`

使用真实数据（padded .npz）：

`python -m src.train --data_npz path/to/data.npz --out_dir runs/exp1`

训练包含三阶段：

- Stage A：监督预训练（`Y` NLL + `K` 的 stepwise supervision）
- Stage B：加入对抗（`D_seq`，WGAN-GP）并保留监督稳定器
- Stage C：加入因果约束（`do` 对齐 / 去偏表示 / 反事实一致性）

### 模型架构（GRU / Transformer）

默认 `GRU`（稳定），可切换为因果 Transformer dynamics：

`python -m src.train --dynamics transformer --tf_n_layers 2 --tf_n_heads 4 --tf_max_seq_len 512 ...`

可选：让 `K_{t+1}` 显式依赖 `Y_t`（更接近 DKT 的反馈更新）：

`python -m src.train --y_in_dynamics ...`

## 因果约束（训练期）

- `do` 对齐：按时间步采样，用 MMD 对齐 `(X, K_t, Y_t)` 的分布（避免二元 `Y` 的“仅均值匹配”退化）；关键参数 `--do_time_sampling {fixed,random,all}`、`--do_num_time_samples`、`--do_actions`。
- 去偏表示：GRL 让 `K_t` 对 `T_t` 的可预测性降低（意图处理 selection bias；不作用于 `K_{t+1}`，避免抹杀 treatment effect）；关键参数 `--w_advT`、`--grl_lambda`。
- 反事实一致性：同一 `(X, eps)` 下改变 `T_t`，要求 `K_{0:t}` 不受影响（主要用于约束因果 masking/实现正确性）；关键参数 `--w_cf`、`--cf_num_time_samples`。

## 评估

干预一致性（均值；可选 `--mmd_xy` 输出基于 `(X,Y)` 的分布距离以反映异质性）：

`python -m src.eval.interventional_eval --ckpt runs/exp1/ckpt_C_ep2.pt --data_npz path/to/data.npz --mmd_xy`

策略模拟（全局均值）：

`python -m src.eval.policy_sim --ckpt runs/exp1/ckpt_C_ep2.pt --data_npz path/to/data.npz --num_actions 3`

策略模拟 + CATE（按 `X[:,x_index]` 分位分组，对比两条常数策略：treated vs control）：

`python -m src.eval.policy_sim --cate --action_control 0 --action_treated 1 --num_groups 2 --x_index 0 --ckpt runs/exp1/ckpt_C_ep2.pt --data_npz path/to/data.npz`

## 代码要点（对应论文可写点）

- `src/model/scm_generator.py`: `rollout()`（支持 `do_t` / `policy` / `eps` 控制），并提供 `dynamics="transformer"` 的 masked self-attention 版本
- `src/model/discriminators.py`: `D_seq`（整段轨迹判别，WGAN-GP）
- `src/model/losses.py`: `L_sup`（TimeGAN-style stepwise）+ `MMD`（do 对齐）+ `GRL`（去偏表示）
- `src/train.py`: Stage A/B/C 训练编排、时间步采样的 `do` 对齐、反事实一致性损失

## 统一入口（SCM + baselines）

新增统一入口脚本：`python -m src.main`，用于在同一套数据接口上训练多种模型：

`python -m src.main --model {scm,rcgan,vae,diffusion,crn} --out_dir runs/NAME [--data_npz path/to/data.npz]`

- GAN：`scm` / `rcgan`（WGAN-GP + supervised term）
- 非对抗：`vae`（ELBO）、`diffusion`（denoising MSE）、`crn`（预测损失 + GRL 去偏）

评估脚本（`src/eval/interventional_eval.py`, `src/eval/policy_sim.py`）已支持读取 `src/train.py` 产生的旧版 SCM checkpoint 和 `src/main.py` 产生的统一 checkpoint。
