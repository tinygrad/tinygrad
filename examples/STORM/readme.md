# Implementation of STORM: Efficient Stochastic Transformer based World Models for Reinforcement Learning

[Paper & OpenReview](https://openreview.net/forum?id=WxnrX42rnS), you may find some useful discussion there.

This repo contains an implementation of STORM. 

Following the **Training and Evaluating Instructions** to reproduce the main results presented in our paper. One may also find **Additional Useful Information** useful when debugging and observing intermediate results. To reproduce the speed metrics mentioned in the paper, please see **Reproducing Speed Metrics**.

## Training and Evaluating Instructions

1. Install the necessary dependencies. Note that we conducted our experiments using `python 3.10`.
    ```shell
    pip install -r requirements.txt
    ```
    Installing `AutoROM.accept-rom-license` may take several minutes.

2. Train the agent.
    ```shell
    chmod +x train.sh
    ./train.sh
    ```

    The `train.sh` file controls the environment and the running name of a training process. 
    ```shell
    env_name=MsPacman
    python -u train.py \
        -n "${env_name}-life_done-wm_2L512D8H-100k-seed1" \
        -seed 1 \
        -config_path "config_files/STORM.yaml" \
        -env_name "ALE/${env_name}-v5" \
        -trajectory_path "trajectory/${env_name}.pkl"
    ```

    - The `env_name` on the first line can be any Atari game, which can be found [here](https://gymnasium.farama.org/environments/atari/).
    
    - `-n` option is the name for the tensorboard logger and checkpoint folder. You can change it to your preference, but we recommend keeping the environment's name first. The tensorboard logging folder is `runs`, and the checkpoint folder is `ckpt`.

    - The `-seed` parameter controls the running seed during the training. We evaluated our method using 5 seeds and report the mean return in Table 1.

    - The `-config_path` points to a YAML file that controls the model's hyperparameters. The configuration in `config_files/STORM.yaml` is the same as in our paper.

    - `-config_path` leads to a yaml file controlling the model's hyperparameters. The configuration in `config_files/STORM.yaml` is the same with our paper.

    - The `-trajectory_path` is only useful when the option `UseDemonstration` in the YAML file is set to `True` (by default it's `False`). This corresponds to the ablation studies in Section 5.3. We provide the precollected trajectories in the `D_TRAJ.7z` file, and you need to decompress it for using (to a `D_TRAJ` folder).
    

3. Evaluate the agent. The evaluation results will be presented in a CSV file located in the `eval_result` folder.
    ```shell
    chmod +x eval.sh
    ./eval.sh
    ```

    The `eval.sh` file controls the environment and the running name when testing an agent.

    ```shell
    env_name=MsPacman
    python -u eval.py \
        -env_name "ALE/${env_name}-v5" \
        -run_name "${env_name}-life_done-wm_2L512D8H-100k-seed1"\
        -config_path "config_files/STORM.yaml" 
    ```

    The `-run_name` option is the same as the `-n` option in `train.sh`. It should be kept the same as in the training script.

## Additional Useful Information
You can use Tensorboard to visualize the training curve and the imagination videos:
```shell
 chmod +x TensorBoard.sh
 ./TensorBoard.sh
 ```


## Reproducing Speed Metrics
To reproduce the speed metrics mentioned in the paper, please consider the following:
- Hardware requirements: NVIDIA GeForce RTX 3090 with a high frequence CPU, we use `11th Gen Intel(R) Core(TM) i9-11900K` in our experiments. Low frequence CPUs may lead to a GPU idle and slow down the traning. To make full use of a powerful GPU, one can traing several agents at the same time on one device.
- Software requiements: `PyTorch>=2.0.0` is required.

We also tested our code on other devices and identified some possible troubleshooting steps:
- Our experiments used bfloat16 to accelerate training. To train on devices that do not support bfloat16, such as the NVIDIA V100, you need to change `torch.bfloat16` to `torch.float16` in both `agents.py` and `sub_models/world_models.py`. Additionally, modify the line `attn = attn.masked_fill(mask == 0, -1e9)` to `attn = attn.masked_fill(mask == 0, -6e4)` to prevent overflow.
- On devices like the NVIDIA A100, using bfloat16 may slow down the training. In this case, you can toggle the `self.use_amp = True` option in both `agents.py` and `sub_models/world_models.py`.

## Code references
We've referenced several other projects during the development of this code:
- [Attention is all you need pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch) For Transformer structure, attention operation, and other building blocks.
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/embeddings.py) For trainable positional encoding.
- [DreamerV3](https://github.com/danijar/dreamerv3) For Symlog loss, layer & kernel configuration in VAE.

## Bibtex

```
@inproceedings{
zhang2023storm,
title={{STORM}: Efficient Stochastic Transformer based World Models for Reinforcement Learning},
author={Weipu Zhang and Gang Wang and Jian Sun and Yetian Yuan and Gao Huang},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=WxnrX42rnS}
}
```
