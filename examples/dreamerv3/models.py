import copy

import networks
import utils

from tinygrad import Tensor, nn, dtypes


class RewardEMA:
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = Tensor([0.05, 0.95]).to(device)
        self.ema_vals = Tensor([0.0, 1.0]).to(device)

    def __call__(self, x):
        flat_x = Tensor.flatten(x.detach())
        x_quantile = utils.quantile(input=flat_x, q=self.range)
        # this should be in-place operation
        self.ema_vals = self.alpha * x_quantile + (1 - self.alpha) * self.ema_vals
        scale = Tensor.maximum(self.ema_vals[1] - self.ema_vals[0], 1.0)
        offset = self.ema_vals[0]
        return offset.detach(), scale.detach()


class WorldModel:
    def __init__(self, obs_space, act_space, step, config):
        self._step = step
        self._config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self.num_actions = (
            int(act_space.n) if hasattr(act_space, "n") else act_space.shape[0]
        )

        self.encoder = networks.MultiEncoder(shapes, **config.encoder)
        self.embed_size = self.encoder.outdim
        self.dynamics = networks.RSSM(
            self.num_actions,
            self.embed_size,
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            config.device,
        )
        self.heads = {}
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.heads["decoder"] = networks.MultiDecoder(
            feat_size, shapes, **config.decoder
        )
        self.heads["reward"] = networks.MLP(
            feat_size,
            (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward",
        )
        self.heads["cont"] = networks.MLP(
            feat_size,
            (),
            config.cont_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist="binary",
            outscale=config.cont_head["outscale"],
            device=config.device,
            name="Cont",
        )
        for name in config.grad_heads:
            assert name in self.heads, name
        lr = config.model_lr
        eps = config.opt_eps
        opt = config.opt
        self._model_opt = {
            "adam": lambda: nn.optim.Adam(self.parameters(), lr=lr, eps=eps),
            "sgd": lambda: nn.optim.SGD(self.parameters(), lr=lr),
            "momentum": lambda: nn.optim.SGD(self.parameters(), lr=lr, momentum=0.9),
        }[opt]()
        self._clip = config.grad_clip
        # other losses are scaled by 1.0.
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"],
        )

    def _train(self, data):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        data = self.preprocess(data)

        embed = self.encoder(data)
        post, prior = self.dynamics.observe(embed, data["action"], data["is_first"])
        kl_free = self._config.kl_free
        dyn_scale = self._config.dyn_scale
        rep_scale = self._config.rep_scale
        kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
            post, prior, kl_free, dyn_scale, rep_scale
        )
        assert kl_loss.shape == embed.shape[:2], kl_loss.shape
        preds = {}
        for name, head in self.heads.items():
            grad_head = name in self._config.grad_heads
            feat = self.dynamics.get_feat(post)
            feat = feat if grad_head else feat.detach()
            pred = head(feat)
            if isinstance(pred, dict):
                preds.update(pred)
            else:
                preds[name] = pred
        losses = {}
        for name, pred in preds.items():
            loss = -pred.log_prob(data[name])
            assert loss.shape == embed.shape[:2], (name, loss.shape, embed.shape[:2])
            losses[name] = loss
        scaled = {
            key: value * self._scales.get(key, 1.0) for key, value in losses.items()
        }
        model_loss = (sum(scaled.values()) + kl_loss).mean()
        metrics = {}
        metrics["model_loss"] = model_loss.item()

        self._model_opt.zero_grad()
        model_loss.backward()
        total_norm = utils.clip_grad_norm_(self.parameters(), self._clip)
        metrics[f"model_grad_norm"] = total_norm.item()
        self._model_opt.step()

        metrics.update({f"{name}_loss": loss.mean().item() for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = dyn_loss.mean().item()
        metrics["rep_loss"] = rep_loss.mean().item()
        metrics["kl"] = kl_value.mean().item()
        metrics["prior_ent"] = self.dynamics.get_dist(prior).entropy().mean().item()
        metrics["post_ent"] = self.dynamics.get_dist(post).entropy().mean().item()
        context = dict(
            embed=embed,
            feat=self.dynamics.get_feat(post),
            kl=kl_value,
            postent=self.dynamics.get_dist(post).entropy(),
        )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    # this function is called during both rollout and training
    def preprocess(self, data):
        data = data.copy()
        data = {
            k: Tensor(v, dtype=dtypes.float32).to(self._config.device)
            for k, v in data.items()
        }
        # onehot encode actions if neccessary
        if "action" in data and len(data["action"].shape) == 2:
            data["action"] = utils.one_hot(
                data["action"].cast(dtypes.int32), self.num_actions
            )
        if "image" in data:
            data["image"] = data["image"].float() / 255.0
        if "discount" in data:
            data["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            data["discount"] = data["discount"].unsqueeze(-1)
        # 'is_first' is necessary to initialize hidden state at training
        assert "is_first" in data
        # 'is_terminal' is necessary to train cont_head
        assert "is_terminal" in data
        data["cont"] = 1.0 - data["is_terminal"]
        return data

    def video_pred(self, data):
        data = self.preprocess(data)
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode
        # observed image is given until 5 steps
        model = Tensor.cat(recon[:, :5], openl, dim=1)
        truth = data["image"][:6]
        error = (model - truth + 1.0) / 2.0

        return Tensor.cat(truth, model, error, dim=2).numpy()

    def state_dict(self):
        models = [self.encoder, self.dynamics, *self.heads.values()]
        state_dict = nn.state.get_state_dict(models)
        return state_dict

    def parameters(self):
        return list(self.state_dict().values())


class ImagBehavior:
    def __init__(self, config, world_model):
        self._config = config
        self._world_model = world_model
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.actor = networks.MLP(
            feat_size,
            (world_model.num_actions,),
            config.actor["layers"],
            config.units,
            config.act,
            config.norm,
            config.actor["dist"],
            "learned",
            config.actor["min_std"],
            config.actor["max_std"],
            absmax=1.0,
            temp=config.actor["temp"],
            unimix_ratio=config.actor["unimix_ratio"],
            outscale=config.actor["outscale"],
            name="Actor",
        )
        self.value = networks.MLP(
            feat_size,
            (255,) if config.critic["dist"] == "symlog_disc" else (),
            config.critic["layers"],
            config.units,
            config.act,
            config.norm,
            config.critic["dist"],
            outscale=config.critic["outscale"],
            device=config.device,
            name="Value",
        )
        if config.critic["slow_target"]:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = utils.Optimizer(
            "actor",
            self.actor_parameters(),
            config.actor["lr"],
            config.actor["eps"],
            config.actor["grad_clip"],
            **kw,
        )
        self._value_opt = utils.Optimizer(
            "value",
            self.value_parameters(),
            config.critic["lr"],
            config.critic["eps"],
            config.critic["grad_clip"],
            **kw,
        )
        if self._config.reward_EMA:
            self.reward_ema = RewardEMA(device=self._config.device)

    def _train(
        self,
        start,
        objective,
    ):
        self._update_slow_target()
        metrics = {}

        with utils.RequiresGrad(self.actor):
            with Tensor.cuda.amp.autocast(self._use_amp):
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._config.imag_horizon
                )
                reward = objective(imag_feat, imag_state, imag_action)
                actor_ent = self.actor(imag_feat).entropy()
                self._world_model.dynamics.get_dist(imag_state).entropy()
                # this target is not scaled by ema or sym_log.
                target, weights, base = self._compute_target(
                    imag_feat, imag_state, reward
                )
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    imag_action,
                    target,
                    weights,
                    base,
                )
                actor_loss -= self._config.actor["entropy"] * actor_ent[:-1, ..., None]
                actor_loss = Tensor.mean(actor_loss)
                metrics.update(mets)
                value_input = imag_feat

        with utils.RequiresGrad(self.value):
            with Tensor.cuda.amp.autocast(self._use_amp):
                value = self.value(value_input[:-1].detach())
                target = Tensor.stack(target, dim=1)
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                value_loss = -value.log_prob(target.detach())
                slow_target = self._slow_value(value_input[:-1].detach())
                if self._config.critic["slow_target"]:
                    value_loss -= value.log_prob(slow_target.mode.detach())
                # (time, batch, 1), (time, batch, 1) -> (1,)
                value_loss = Tensor.mean(weights[:-1] * value_loss[:, :, None])

        metrics.update(utils.tensorstats(value.mode, "value"))
        metrics.update(utils.tensorstats(target, "target"))
        metrics.update(utils.tensorstats(reward, "imag_reward"))
        if self._config.actor["dist"] in ["onehot"]:
            metrics.update(
                utils.tensorstats(
                    Tensor.argmax(imag_action, dim=-1).float(), "imag_action"
                )
            )
        else:
            metrics.update(utils.tensorstats(imag_action, "imag_action"))
        metrics["actor_entropy"] = Tensor.mean(actor_ent).numpy()
        metrics.update(self._actor_opt(actor_loss, self.actor_parameters()))
        metrics.update(self._value_opt(value_loss, self.value_parameters()))
        return imag_feat, imag_state, imag_action, weights, metrics

    def _imagine(self, start, policy, horizon):
        dynamics = self._world_model.dynamics

        def flatten(x):
            return x.reshape([-1] + list(x.shape[2:]))

        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach()
            action = policy(inp).sample()
            succ = dynamics.img_step(state, action)
            return succ, feat, action

        succ, feats, actions = utils.static_scan(
            step, [Tensor.arange(horizon)], (start, None, None)
        )
        states = {k: Tensor.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}

        return feats, states, actions

    def _compute_target(self, imag_feat, imag_state, reward):
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        else:
            discount = self._config.discount * Tensor.ones_like(reward)
        value = self.value(imag_feat).mode
        target = utils.lambda_return(
            reward[1:],
            value[:-1],
            discount[1:],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        weights = Tensor.cumprod(
            Tensor.cat([Tensor.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]

    def _compute_actor_loss(
        self,
        imag_feat,
        imag_action,
        target,
        weights,
        base,
    ):
        metrics = {}
        inp = imag_feat.detach()
        policy = self.actor(inp)
        # Q-val for actor is not transformed using symlog
        target = Tensor.stack(target, dim=1)
        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(utils.tensorstats(normed_target, "normed_target"))
            metrics["EMA_005"] = self.reward_ema.ema_vals[0].numpy()
            metrics["EMA_095"] = self.reward_ema.ema_vals[1].numpy()

        if self._config.imag_gradient == "dynamics":
            actor_target = adv
        elif self._config.imag_gradient == "reinforce":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode).detach()
            )
        elif self._config.imag_gradient == "both":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode).detach()
            )
            mix = self._config.imag_gradient_mix
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        actor_loss = -weights[:-1] * actor_target
        return actor_loss, metrics

    def _update_slow_target(self):
        if self._config.critic["slow_target"]:
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1

    def actor_parameters(self):
        return nn.state.get_parameters(self.actor)

    def value_parameters(self):
        return nn.state.get_parameters(self.value)
