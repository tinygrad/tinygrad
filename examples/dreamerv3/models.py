import copy

import distributions
import networks
import utils

from tinygrad import Tensor, TinyJit, dtypes, nn


class RewardEMA:
    """running mean and std"""

    def __init__(self, alpha=1e-2):
        self.alpha = alpha
        self.range = Tensor([0.05, 0.95])
        self.ema_vals = Tensor([0.0, 1.0])

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
        self.num_actions = int(act_space.n) if hasattr(act_space, "n") else act_space.shape[0]

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
        )
        self.heads = {}
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.heads["decoder"] = networks.MultiDecoder(feat_size, shapes, **config.decoder)
        self.heads["reward"] = networks.MLP(
            feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
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
            name="Cont",
        )
        for name in config.grad_heads:
            assert name in self.heads, name
        # other losses are scaled by 1.0.
        self._scales = dict(reward=config.reward_head["loss_scale"], cont=config.cont_head["loss_scale"])

        self.opt = utils.Optimizer("model", self.parameters(), config.model_lr, config.opt_eps, config.grad_clip, config.opt)

    def preprocess(self, data):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        data = data.copy()
        data = {k: Tensor(v, dtype=dtypes.float32) for k, v in data.items()}
        # onehot encode actions if neccessary
        if "action" in data and len(data["action"].shape) == 2:
            data["action"] = Tensor.one_hot(data["action"].cast(dtypes.int32), self.num_actions)
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

    @staticmethod
    @TinyJit
    def _train(model: "WorldModel", **data):
        embed = model.encoder(data)
        post, prior = model.dynamics.observe(embed, data["action"], data["is_first"])
        kl_free = model._config.kl_free
        dyn_scale = model._config.dyn_scale
        rep_scale = model._config.rep_scale
        kl_loss, kl_value, dyn_loss, rep_loss = model.dynamics.kl_loss(post, prior, kl_free, dyn_scale, rep_scale)
        assert kl_loss.shape == embed.shape[:2], kl_loss.shape
        preds = {}
        for name, head in model.heads.items():
            grad_head = name in model._config.grad_heads
            feat = model.dynamics.get_feat(post)
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
        scaled = {key: value * model._scales.get(key, 1.0) for key, value in losses.items()}
        model_loss = (sum(scaled.values()) + kl_loss).mean()
        metrics = {}
        metrics["model_loss"] = model_loss

        model.opt.zero_grad()
        model_loss.backward()
        metrics.update(model.opt.step())

        metrics.update({f"{name}_loss": loss.mean() for name, loss in losses.items()})
        metrics["dyn_loss"] = dyn_loss.mean()
        metrics["rep_loss"] = rep_loss.mean()
        metrics["kl"] = kl_value.mean()
        metrics["prior_ent"] = model.dynamics.get_dist(prior).entropy().mean()
        metrics["post_ent"] = model.dynamics.get_dist(post).entropy().mean()
        context = dict(
            embed=embed,
            feat=model.dynamics.get_feat(post),
            kl=kl_value,
            postent=model.dynamics.get_dist(post).entropy(),
        )
        post = {k: v.detach().realize() for k, v in post.items()}
        context = {k: v.detach().realize() for k, v in context.items()}
        metrics = {k: v.realize() for k, v in metrics.items()}
        return post, context, metrics

    def train(self, data):
        with Tensor.train():
            data = self.preprocess(data)
            return self._train(self, **data)

    @staticmethod
    @TinyJit
    def _video_pred(model: "WorldModel", **data):
        embed = model.encoder(data)

        states, _ = model.dynamics.observe(embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5])
        recon = model.heads["decoder"](model.dynamics.get_feat(states))["image"].mode[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = model.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        openl = model.heads["decoder"](model.dynamics.get_feat(prior))["image"].mode
        model = Tensor.cat(recon[:, :5], openl, dim=1)
        truth = data["image"][:6]
        error = (model - truth + 1.0) / 2.0
        return Tensor.cat(truth, model, error, dim=2).realize()

    def video_pred(self, data):
        with Tensor.train(False):
            data = self.preprocess(data)
            pred = self._video_pred(self, **data)
        return pred

    def parameters(self):
        models = [self.encoder, self.dynamics, *self.heads.values()]
        return nn.state.get_parameters(models)


class ActorCritic:
    def __init__(self, config, world_model: WorldModel):
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
            "learned" if config.actor["dist"] == "normal" else 0.0,
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
            name="Value",
        )
        if config.critic["slow_target"]:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        self._actor_opt = utils.Optimizer(
            "actor", self.actor_parameters(), config.actor["lr"], config.actor["eps"], config.actor["grad_clip"], config.opt
        )
        self._value_opt = utils.Optimizer(
            "value", self.value_parameters(), config.critic["lr"], config.critic["eps"], config.critic["grad_clip"], config.opt
        )
        if self._config.reward_EMA:
            self.reward_ema = RewardEMA()

    @staticmethod
    @TinyJit
    def _train(actor_critic: "ActorCritic", imag_feat, imag_action, reward, **imag_state):
        metrics = {}
        actor_ent = actor_critic.actor(imag_feat).entropy()
        # this target is not scaled by ema or sym_log.
        target, weights, base = actor_critic._compute_target(imag_feat, imag_state, reward)
        actor_loss, mets = actor_critic._compute_actor_loss(imag_feat, imag_action, target.detach(), weights.detach(), base.detach())
        actor_loss = actor_loss - actor_critic._config.actor["entropy"] * actor_ent[:-1, ..., None]
        actor_loss = Tensor.mean(actor_loss)
        metrics.update(mets)

        value = actor_critic.value(imag_feat[:-1])
        target = target.squeeze(-1).transpose().detach()
        # (time, batch, 1), (time, batch, 1) -> (time, batch)
        value_loss = -value.log_prob(target)
        if actor_critic._config.critic["slow_target"]:
            slow_target = actor_critic._slow_value(imag_feat[:-1]).mode
            slow_target = slow_target.squeeze(-1).detach()
            value_loss = value_loss - value.log_prob(slow_target)
        # (time, batch, 1), (time, batch, 1) -> (1,)
        value_loss = Tensor.mean(weights[:-1] * value_loss[:, :, None])

        metrics.update(utils.tensorstats(value.mode, "value"))
        metrics.update(utils.tensorstats(target, "target"))
        metrics.update(utils.tensorstats(reward, "imag_reward"))
        if actor_critic._config.actor["dist"] == "onehot":
            metrics.update(utils.tensorstats(Tensor.argmax(imag_action, -1).float(), "imag_action"))
        else:
            metrics.update(utils.tensorstats(imag_action, "imag_action"))
        metrics["actor_entropy"] = Tensor.mean(actor_ent)

        metrics["actor_loss"] = actor_loss
        metrics["value_loss"] = value_loss

        actor_critic._actor_opt.zero_grad()
        actor_loss.backward()
        metrics.update(actor_critic._actor_opt.step())

        actor_critic._value_opt.zero_grad()
        value_loss.backward()
        metrics.update(actor_critic._value_opt.step())

        metrics = {k: v.realize() for k, v in metrics.items()}
        return imag_feat, imag_state, imag_action, weights, metrics

    @staticmethod
    @TinyJit
    def _imagine(actor_critic: "ActorCritic", **start):
        dynamics = actor_critic._world_model.dynamics

        def flatten(x):
            return x.reshape([-1] + list(x.shape[2:]))

        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach()
            action = actor_critic.actor(inp).sample()
            succ = dynamics.img_step(state, action)
            return succ, feat, action

        succ, feats, actions = utils.static_scan(step, [Tensor.arange(actor_critic._config.imag_horizon)], (start, None, None))
        states = {k: Tensor.cat(start[k][None], v[:-1], dim=0) for k, v in succ.items()}

        feats = feats.realize()
        actions = actions.realize()
        states = {k: v.realize() for k, v in states.items()}

        return feats, states, actions

    def _compute_target(self, imag_feat, imag_state, reward):
        reward = reward.unsqueeze(-1)
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        else:
            discount = self._config.discount * Tensor.ones_like(reward)
        # realize value here for metrics to work
        value = self.value(imag_feat).mode.realize()
        target = utils.lambda_return(reward[1:], value[:-1], discount[1:], bootstrap=value[-1], lambda_=self._config.discount_lambda, axis=0)
        weights = utils.cumprod(Tensor.cat(Tensor.ones_like(discount[:1]), discount[:-1], dim=0), 0).detach()
        return target, weights, value[:-1]

    def _compute_actor_loss(self, imag_feat, imag_action, target, weights, base):
        metrics = {}
        policy = self.actor(imag_feat)
        # Q-val for actor is not transformed using symlog
        target = Tensor.stack(target, dim=1)
        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(utils.tensorstats(normed_target, "normed_target"))
            metrics["EMA_005"] = self.reward_ema.ema_vals[0]
            metrics["EMA_095"] = self.reward_ema.ema_vals[1]

        if self._config.imag_gradient == "dynamics":
            actor_target = adv
        elif self._config.imag_gradient == "reinforce":
            actor_target = policy.log_prob(imag_action)[:-1][:, :, None] * (target - self.value(imag_feat[:-1]).mode).detach()
        elif self._config.imag_gradient == "both":
            actor_target = policy.log_prob(imag_action)[:-1][:, :, None] * (target - self.value(imag_feat[:-1]).mode).detach()
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
                for s, d in zip(self.value_parameters(), self.slow_value_parameters()):
                    d *= 1 - mix
                    d += mix * s.detach()
            self._updates += 1

    def train(self, start, objective):
        self._update_slow_target()
        imag_feat, imag_state, imag_action = self._imagine(self, **start)
        imag_feat = imag_feat.detach().contiguous().realize()
        imag_state = {k: v.detach().contiguous().realize() for k, v in imag_state.items()}
        imag_action = imag_action.contiguous().realize()
        imag_reward = objective(imag_feat, imag_state, imag_action).detach().contiguous().realize()
        return self._train(self, imag_feat, imag_action, imag_reward, **imag_state)

    def actor_parameters(self):
        return nn.state.get_parameters(self.actor)

    def value_parameters(self):
        return nn.state.get_parameters(self.value)

    def slow_value_parameters(self):
        return nn.state.get_parameters(self._slow_value)

    @staticmethod
    @TinyJit
    def _train_policy(actor_critic: "ActorCritic", image, is_first, action, **latent):
        obs = {"image": image, "is_first": is_first}
        embed = actor_critic._world_model.encoder({k: v[:, None] for k, v in obs.items()})[:, 0]
        latent, _ = actor_critic._world_model.dynamics.obs_step(latent, action, embed, obs["is_first"])
        feat = actor_critic._world_model.dynamics.get_feat(latent)
        actor = actor_critic.actor(feat)
        action = actor.mode
        action = action.detach().realize()
        logprob = actor.log_prob(action).detach().realize()
        policy_output = {"action": action, "logprob": logprob}
        latent = {k: v.detach().realize() for k, v in latent.items()}
        state = (latent, action)
        return policy_output, state

    @staticmethod
    @TinyJit
    def _train_policy_expl(actor_critic: "ActorCritic", image, is_first, action, **latent):
        obs = {"image": image, "is_first": is_first}
        embed = actor_critic._world_model.encoder({k: v[:, None] for k, v in obs.items()})[:, 0]
        latent, _ = actor_critic._world_model.dynamics.obs_step(latent, action, embed, obs["is_first"])
        feat = actor_critic._world_model.dynamics.get_feat(latent)
        actor = actor_critic.actor(feat)
        action = actor.sample()
        action = action.detach().realize()
        logprob = actor.log_prob(action).detach().realize()
        policy_output = {"action": action, "logprob": logprob}
        latent = {k: v.detach().realize() for k, v in latent.items()}
        state = (latent, action)
        return policy_output, state

    @staticmethod
    @TinyJit
    def _eval_policy(actor_critic: "ActorCritic", image, is_first, action, **latent):
        obs = {"image": image, "is_first": is_first}
        embed = actor_critic._world_model.encoder({k: v[:, None] for k, v in obs.items()})[:, 0]
        latent, _ = actor_critic._world_model.dynamics.obs_step(latent, action, embed, obs["is_first"])
        feat = actor_critic._world_model.dynamics.get_feat(latent)
        actor = actor_critic.actor(feat)
        action = actor.sample()
        action = action.detach().realize()
        logprob = actor.log_prob(action).detach().realize()
        policy_output = {"action": action, "logprob": logprob}
        latent = {k: v.detach().realize() for k, v in latent.items()}
        state = (latent, action)
        return policy_output, state

    def policy(self, obs, state, training: bool = False, explore: bool = False):
        with Tensor.train(False):
            obs = self._world_model.preprocess(obs)
            B = obs["is_first"].shape[0]
            if state is None:
                latent = self._world_model.dynamics.initial(B)
                action = Tensor.zeros([B, self._world_model.num_actions])
                state = (latent, action)
            else:
                latent, action = state
            latent = {k: v.contiguous().realize() for k, v in latent.items()}
            action = action.contiguous().realize()
            image, is_first = obs["image"], obs["is_first"]
            if training:
                if explore:
                    policy_fn = self._train_policy_expl
                else:
                    policy_fn = self._train_policy
            else:
                policy_fn = self._eval_policy
            policy_output, state = policy_fn(self, image, is_first, action, **latent)
            return policy_output, state


def random_agent(config, act_space):
    if config.actor["dist"] == "onehot":
        random_actor = distributions.OneHotCategorical(Tensor.zeros(int(act_space.n)).repeat((config.num_envs, 1)))
    else:
        random_actor = distributions.Independent(
            distributions.Uniform(
                Tensor(act_space.low).repeat((config.num_envs, 1)),
                Tensor(act_space.high).repeat((config.num_envs, 1)),
            ),
            1,
        )
    def random_policy(o, s):
        action = random_actor.sample()
        logprob = random_actor.log_prob(action)
        return {"action": action, "logprob": logprob}, None
    
    return random_policy
