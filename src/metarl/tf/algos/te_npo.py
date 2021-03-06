"""Natural Policy Optimization with Task Embeddings."""
# pylint: disable=too-many-lines
import akro
from dowel import Histogram, logger, tabular
import numpy as np
import scipy.stats
import tensorflow as tf

from metarl import InOutSpec, log_performance, TrajectoryBatch
from metarl.misc import tensor_utils as np_tensor_utils
from metarl.np.algos import RLAlgorithm
from metarl.sampler import LocalSampler
from metarl.tf.embeddings import StochasticEncoder
from metarl.tf.misc.tensor_utils import center_advs
from metarl.tf.misc.tensor_utils import compile_function
from metarl.tf.misc.tensor_utils import compute_advantages
from metarl.tf.misc.tensor_utils import concat_tensor_list
from metarl.tf.misc.tensor_utils import discounted_returns
from metarl.tf.misc.tensor_utils import flatten_inputs
from metarl.tf.misc.tensor_utils import graph_inputs
from metarl.tf.misc.tensor_utils import pad_tensor
from metarl.tf.misc.tensor_utils import pad_tensor_dict
from metarl.tf.misc.tensor_utils import pad_tensor_n
from metarl.tf.misc.tensor_utils import positive_advs
from metarl.tf.misc.tensor_utils import stack_tensor_dict_list
from metarl.tf.optimizers import LbfgsOptimizer
from metarl.tf.policies import TaskEmbeddingPolicy


class TENPO(RLAlgorithm):
    """Natural Policy Optimization with Task Embeddings.

    See https://karolhausman.github.io/pdf/hausman17nips-ws2.pdf for algorithm
    reference.

    Args:
        env_spec (metarl.envs.EnvSpec): Environment specification.
        policy (metarl.tf.policies.TaskEmbeddingPolicy): Policy.
        baseline (metarl.tf.baselines.Baseline): The baseline.
        scope (str): Scope for identifying the algorithm.
            Must be specified if running multiple algorithms
            simultaneously, each using different environments
            and policies.
        max_path_length (int): Maximum length of a single rollout.
        discount (float): Discount.
        gae_lambda (float): Lambda used for generalized advantage
            estimation.
        center_adv (bool): Whether to rescale the advantages
            so that they have mean 0 and standard deviation 1.
        positive_adv (bool): Whether to shift the advantages
            so that they are always positive. When used in
            conjunction with center_adv the advantages will be
            standardized before shifting.
        fixed_horizon (bool): Whether to fix horizon.
        pg_loss (str): A string from: 'vanilla', 'surrogate',
            'surrogate_clip'. The type of loss functions to use.
        lr_clip_range (float): The limit on the likelihood ratio between
            policies, as in PPO.
        max_kl_step (float): The maximum KL divergence between old and new
            policies, as in TRPO.
        optimizer (object): The optimizer of the algorithm. Should be the
            optimizers in metarl.tf.optimizers.
        optimizer_args (dict): The arguments of the optimizer.
        policy_ent_coeff (float): The coefficient of the policy entropy.
            Setting it to zero would mean no entropy regularization.
        encoder_ent_coeff (float): The coefficient of the policy encoder
            entropy. Setting it to zero would mean no entropy regularization.
        use_softplus_entropy (bool): Whether to estimate the softmax
            distribution of the entropy to prevent the entropy from being
            negative.
        use_neg_logli_entropy (bool): Whether to estimate the entropy as the
            negative log likelihood of the action.
        stop_entropy_gradient (bool): Whether to stop the entropy gradient.
        stop_ce_gradient (bool): Whether to stop the cross entropy gradient.
        entropy_method (str): A string from: 'max', 'regularized',
            'no_entropy'. The type of entropy method to use. 'max' adds the
            dense entropy to the reward for each time step. 'regularized' adds
            the mean entropy to the surrogate objective. See
            https://arxiv.org/abs/1805.00909 for more details.
        flatten_input (bool): Whether to flatten input along the observation
            dimension. If True, for example, an observation with shape (2, 4)
            will be flattened to 8.
        inference (metarl.tf.embeddings.StochasticEncoder): A encoder
            that infers the task embedding from trajectory.
        inference_optimizer (object): The optimizer of the inference. Should be
            an optimizer in metarl.tf.optimizers.
        inference_optimizer_args (dict): The arguments of the inference
            optimizer.
        inference_ce_coeff (float): The coefficient of the cross entropy of
            task embeddings inferred from task one-hot and trajectory. This is
            effectively the coefficient of log-prob of inference.
        name (str): The name of the algorithm.

    Note:
        sane defaults for entropy configuration:
            - entropy_method='max', center_adv=False, stop_gradient=True
              (center_adv normalizes the advantages tensor, which will
              significantly alleviate the effect of entropy. It is also
              recommended to turn off entropy gradient so that the agent
              will focus on high-entropy actions instead of increasing the
              variance of the distribution.)
            - entropy_method='regularized', stop_gradient=False,
              use_neg_logli_entropy=False

    """

    def __init__(self,
                 env_spec,
                 policy,
                 baseline,
                 scope=None,
                 max_path_length=500,
                 discount=0.99,
                 gae_lambda=1,
                 center_adv=True,
                 positive_adv=False,
                 fixed_horizon=False,
                 pg_loss='surrogate',
                 lr_clip_range=0.01,
                 max_kl_step=0.01,
                 optimizer=None,
                 optimizer_args=None,
                 policy_ent_coeff=0.0,
                 encoder_ent_coeff=0.0,
                 use_softplus_entropy=False,
                 use_neg_logli_entropy=False,
                 stop_entropy_gradient=False,
                 stop_ce_gradient=False,
                 entropy_method='no_entropy',
                 flatten_input=True,
                 inference=None,
                 inference_optimizer=None,
                 inference_optimizer_args=None,
                 inference_ce_coeff=0.0,
                 name='NPOTaskEmbedding'):
        assert isinstance(policy, TaskEmbeddingPolicy)
        assert isinstance(inference, StochasticEncoder)

        self.policy = policy
        self.scope = scope
        self.max_path_length = max_path_length

        self._env_spec = env_spec
        self._baseline = baseline
        self._discount = discount
        self._gae_lambda = gae_lambda
        self._center_adv = center_adv
        self._positive_adv = positive_adv
        self._fixed_horizon = fixed_horizon
        self._flatten_input = flatten_input
        self._name = name
        self._name_scope = tf.name_scope(self._name)
        self._old_policy = policy.clone('old_policy')
        self._use_softplus_entropy = use_softplus_entropy
        self._use_neg_logli_entropy = use_neg_logli_entropy
        self._stop_entropy_gradient = stop_entropy_gradient
        self._stop_ce_gradient = stop_ce_gradient
        self._pg_loss = pg_loss

        optimizer, optimizer_args = self._build_optimizer(
            optimizer, optimizer_args)
        inference_opt, inference_opt_args = self._build_inference_optimizer(
            inference_optimizer, inference_optimizer_args)

        self._check_entropy_configuration(entropy_method, center_adv,
                                          stop_entropy_gradient,
                                          use_neg_logli_entropy,
                                          policy_ent_coeff)

        if pg_loss not in ['vanilla', 'surrogate', 'surrogate_clip']:
            raise ValueError('Invalid pg_loss')

        with self._name_scope:
            self._optimizer = optimizer(**optimizer_args)
            self._lr_clip_range = float(lr_clip_range)
            self._max_kl_step = float(max_kl_step)
            self._policy_ent_coeff = float(policy_ent_coeff)

            self._inference = inference
            self._old_inference = inference.clone('old_inference')
            self.inference_ce_coeff = float(inference_ce_coeff)
            self.inference_optimizer = inference_opt(**inference_opt_args)
            self.encoder_ent_coeff = encoder_ent_coeff

        self._f_rewards = None
        self._f_returns = None
        self._f_policy_kl = None
        self._f_policy_entropy = None
        self._f_encoder_kl = None
        self._f_encoder_entropy = None
        self._f_task_entropies = None
        self._f_inference_ce = None

        self.sampler_cls = LocalSampler

        self.init_opt()

    def init_opt(self):
        """Initialize optimizater.

        Raises:
            NotImplementedError: Raise if the policy is recurrent.

        """
        # Input variables
        (pol_loss_inputs, pol_opt_inputs, infer_loss_inputs,
         infer_opt_inputs) = self._build_inputs()

        self._policy_opt_inputs = pol_opt_inputs
        self._inference_opt_inputs = infer_opt_inputs

        # Jointly optimize policy and encoder network
        pol_loss, pol_kl, _ = self._build_policy_loss(pol_loss_inputs)
        self._optimizer.update_opt(loss=pol_loss,
                                   target=self.policy,
                                   leq_constraint=(pol_kl, self._max_kl_step),
                                   inputs=flatten_inputs(
                                       self._policy_opt_inputs),
                                   constraint_name='mean_kl')

        # Optimize inference distribution separately (supervised learning)
        infer_loss, _ = self._build_inference_loss(infer_loss_inputs)
        self.inference_optimizer.update_opt(loss=infer_loss,
                                            target=self._inference,
                                            inputs=flatten_inputs(
                                                self._inference_opt_inputs))

    def train(self, runner):
        """Obtain samplers and start actual training for each epoch.

        Args:
            runner (LocalRunner): LocalRunner is passed to give algorithm
                the access to runner.step_epochs(), which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """
        last_return = None

        for _ in runner.step_epochs():
            runner.step_path = runner.obtain_samples(runner.step_itr)
            last_return = self.train_once(runner.step_itr, runner.step_path)
            runner.step_itr += 1

        return last_return

    def train_once(self, itr, paths):
        """Perform one step of policy optimization given one batch of samples.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths.

        Returns:
            numpy.float64: Average return.

        """
        samples_data = self.paths_to_tensors(itr, paths)
        logger.log('Optimizing policy...')
        self.optimize_policy(itr, samples_data)
        return samples_data['average_return']

    def optimize_policy(self, itr, samples_data):
        """Optimize policy.

        Args:
            itr (int): Iteration number.
            samples_data (dict): Processed sample data.
                See process_samples() for details.

        """
        del itr

        policy_opt_input_values = self._policy_opt_input_values(samples_data)
        inference_opt_input_values = self._inference_opt_input_values(
            samples_data)

        self._train_policy_and_encoder_networks(policy_opt_input_values)
        self._train_inference_network(inference_opt_input_values)

        paths = samples_data['paths']
        self.evaluate(policy_opt_input_values, samples_data)
        self.visualize_distribution()

        logger.log('Fitting baseline...')
        self._baseline.fit(paths)

        self._old_policy.model.parameters = self.policy.model.parameters
        self._old_policy.encoder.model.parameters = (
            self.policy.encoder.model.parameters)
        self._old_inference.model.parameters = self._inference.model.parameters

    def paths_to_tensors(self, itr, paths):
        # pylint: disable=too-many-statements
        """Return processed sample data based on the collected paths.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths.

        Returns:
            dict: Processed sample data, with key
                * observations: (numpy.ndarray)
                * actions: (numpy.ndarray)
                * rewards: (numpy.ndarray)
                * baselines: (numpy.ndarray)
                * returns: (numpy.ndarray)
                * valids: (numpy.ndarray)
                * agent_infos: (dict)
                * env_infos: (dict)
                * paths: (list[dict])
                * average_return: (numpy.float64)

        """
        baselines = []
        returns = []
        total_steps = 0

        max_path_length = self.max_path_length

        undiscounted_returns = log_performance(
            itr,
            TrajectoryBatch.from_trajectory_list(self._env_spec, paths),
            discount=self._discount)

        def _extract_latent_infos(infos):
            """Extract and pack latent infos from dict.

            Args:
                infos (dict): A dict that contains latent infos with key
                    prefixed by 'latent_'.

            Returns:
                dict: A dict of latent infos.

            """
            latent_infos = dict()
            for k, v in infos.items():
                if k.startswith('latent_'):
                    latent_infos[k[7:]] = v
            return latent_infos

        if self._flatten_input:
            paths = [
                dict(observations=(self._env_spec.observation_space.flatten_n(
                    path['observations'])),
                     tasks=self.policy.task_space.flatten_n(
                         path['env_infos']['task_onehot']),
                     latents=path['agent_infos']['latent'],
                     actions=self._env_spec.action_space.flatten_n(
                         path['actions']),
                     rewards=path['rewards'],
                     env_infos=path['env_infos'],
                     agent_infos=path['agent_infos'],
                     latent_infos=_extract_latent_infos(path['agent_infos']),
                     dones=path['dones']) for path in paths
            ]
        else:
            paths = [
                dict(observations=path['observations'],
                     tasks=path['env_infos']['task_onehot'],
                     latenst=path['agent_infos']['latent'],
                     actions=(self._env_spec.action_space.flatten_n(
                         path['actions'])),
                     rewards=path['rewards'],
                     env_infos=path['env_infos'],
                     agent_infos=path['agent_infos'],
                     latent_infos=_extract_latent_infos(path['agent_infos']),
                     dones=path['dones']) for path in paths
            ]

        all_path_baselines = [self._baseline.predict(path) for path in paths]

        for idx, path in enumerate(paths):
            total_steps += len(path['rewards'])
            path_baselines = np.append(all_path_baselines[idx], 0)
            deltas = (path['rewards'] + self._discount * path_baselines[1:] -
                      path_baselines[:-1])
            path['advantages'] = np_tensor_utils.discount_cumsum(
                deltas, self._discount * self._gae_lambda)
            path['deltas'] = deltas

        for idx, path in enumerate(paths):
            # baselines
            path['baselines'] = all_path_baselines[idx]
            baselines.append(path['baselines'])

            # returns
            path['returns'] = np_tensor_utils.discount_cumsum(
                path['rewards'], self._discount)
            returns.append(path['returns'])

        # calculate inference trajectories samples
        for idx, path in enumerate(paths):
            # - Calculate a forward-looking sliding window.
            # - If step_space has shape (n, d), then trajs will have shape
            #   (n, window, d)
            # - The length of the sliding window is determined by the
            #   trajectory inference spec. We smear the last few elements to
            #   preserve the time dimension.
            # - Only observation is used for a single step.
            #   Alternatively, stacked [observation, action] can be used for
            #   in harder tasks.
            obs = pad_tensor(path['observations'], max_path_length)
            obs_flat = self._env_spec.observation_space.flatten_n(obs)
            steps = obs_flat
            window = self._inference.spec.input_space.shape[0]
            traj = np_tensor_utils.sliding_window(steps, window, smear=True)
            traj_flat = self._inference.spec.input_space.flatten_n(traj)
            path['trajectories'] = traj_flat

            _, traj_info = self._inference.get_latents(traj_flat)
            path['trajectory_infos'] = traj_info

        # make all paths the same length
        obs = [path['observations'] for path in paths]
        obs = pad_tensor_n(obs, max_path_length)

        actions = [path['actions'] for path in paths]
        actions = pad_tensor_n(actions, max_path_length)

        tasks = [path['tasks'] for path in paths]
        tasks = pad_tensor_n(tasks, max_path_length)

        latents = [path['latents'] for path in paths]
        latents = pad_tensor_n(latents, max_path_length)

        rewards = [path['rewards'] for path in paths]
        rewards = pad_tensor_n(rewards, max_path_length)

        returns = [path['returns'] for path in paths]
        returns = pad_tensor_n(returns, max_path_length)

        baselines = pad_tensor_n(baselines, max_path_length)

        trajectories = np.stack([path['trajectories'] for path in paths])

        agent_infos = [path['agent_infos'] for path in paths]
        agent_infos = stack_tensor_dict_list(
            [pad_tensor_dict(p, max_path_length) for p in agent_infos])

        latent_infos = [path['latent_infos'] for path in paths]
        latent_infos = stack_tensor_dict_list(
            [pad_tensor_dict(p, max_path_length) for p in latent_infos])

        trajectory_infos = [path['trajectory_infos'] for path in paths]
        trajectory_infos = stack_tensor_dict_list(
            [pad_tensor_dict(p, max_path_length) for p in trajectory_infos])

        env_infos = [path['env_infos'] for path in paths]
        env_infos = stack_tensor_dict_list(
            [pad_tensor_dict(p, max_path_length) for p in env_infos])

        valids = [np.ones_like(path['returns']) for path in paths]
        valids = pad_tensor_n(valids, max_path_length)

        lengths = np.asarray([v.sum() for v in valids])

        samples_data = dict(
            observations=obs,
            actions=actions,
            tasks=tasks,
            latents=latents,
            trajectories=trajectories,
            rewards=rewards,
            baselines=baselines,
            returns=returns,
            valids=valids,
            lengths=lengths,
            agent_infos=agent_infos,
            env_infos=env_infos,
            latent_infos=latent_infos,
            trajectory_infos=trajectory_infos,
            paths=paths,
            average_return=np.mean(undiscounted_returns),
        )

        return samples_data

    def _build_optimizer(self, optimizer, optimizer_args):
        """Build up optimizer for policy.

        Args:
            optimizer (obj): Policy optimizer. Should be one of the optimizers
                in metarl.tf.optimizers.
            optimizer_args (dict): The arguments of the optimizer.

        Returns:
            obj: Policy optimizer. Should be one of the optimizers
                in metarl.tf.optimizers.
            dict: The arguments of the optimizer.

        """
        # pylint: disable=no-self-use
        if optimizer is None:
            optimizer = LbfgsOptimizer
        if optimizer_args is None:
            optimizer_args = dict()
        return optimizer, optimizer_args

    def _build_inference_optimizer(self, optimizer, optimizer_args):
        """Build up optimizer for inference.

        Args:
            optimizer (obj): Policy optimizer. Should be one of the optimizers
                in metarl.tf.optimizers.
            optimizer_args (dict): The arguments of the optimizer.

        Returns:
            obj: Policy optimizer. Should be one of the optimizers
                in metarl.tf.optimizers.
            dict: The arguments of the optimizer.

        """
        return self._build_optimizer(optimizer, optimizer_args)

    def _build_inputs(self):
        """Build input variables.

        Returns:
            namedtuple: Collection of variables to compute policy loss.
            namedtuple: Collection of variables to do policy optimization.
            namedtuple: Collection of variables to compute inference loss.
            namedtuple: Collection of variables to do inference optimization.

        """
        # pylint: disable=too-many-statements
        observation_space = self.policy.observation_space
        action_space = self.policy.action_space
        task_space = self.policy.task_space
        latent_space = self.policy.latent_space
        trajectory_space = self._inference.spec.input_space

        with tf.name_scope('inputs'):
            if self._flatten_input:
                obs_var = tf.compat.v1.placeholder(
                    tf.float32,
                    shape=[None, None, observation_space.flat_dim],
                    name='obs')
                task_var = tf.compat.v1.placeholder(
                    tf.float32,
                    shape=[None, None, task_space.flat_dim],
                    name='task')
                trajectory_var = tf.compat.v1.placeholder(
                    tf.float32, shape=[None, None, trajectory_space.flat_dim])
                latent_var = tf.compat.v1.placeholder(
                    tf.float32, shape=[None, None, latent_space.flat_dim])
            else:
                obs_var = observation_space.to_tf_placeholder(name='obs',
                                                              batch_dims=2)
                task_var = task_space.to_tf_placeholder(name='task',
                                                        batch_dims=2)
                trajectory_var = trajectory_space.to_tf_placeholder(
                    name='trajectory', batch_dims=2)
                latent_var = latent_space.to_tf_placeholder(name='latent',
                                                            batch_dims=2)
            action_var = action_space.to_tf_placeholder(name='action',
                                                        batch_dims=2)
            reward_var = tf.compat.v1.placeholder(tf.float32,
                                                  shape=[None, None],
                                                  name='reward')
            baseline_var = tf.compat.v1.placeholder(tf.float32,
                                                    shape=[None, None],
                                                    name='baseline')

            valid_var = tf.compat.v1.placeholder(tf.float32,
                                                 shape=[None, None],
                                                 name='valid')

            # Policy state (for RNNs)
            policy_state_info_vars = {
                k: tf.compat.v1.placeholder(tf.float32,
                                            shape=[None] * 2 + list(shape),
                                            name=k)
                for k, shape in self.policy.state_info_specs
            }
            policy_state_info_vars_list = [
                policy_state_info_vars[k] for k in self.policy.state_info_keys
            ]

            # Encoder state (for RNNs)
            embed_state_info_vars = {
                k: tf.compat.v1.placeholder(tf.float32,
                                            shape=[None] * 2 + list(shape),
                                            name='embed_%s' % k)
                for k, shape in self.policy.encoder.state_info_specs
            }
            embed_state_info_vars_list = [
                embed_state_info_vars[k]
                for k in self.policy.encoder.state_info_keys
            ]

            # Inference distribution state (for RNNs)
            infer_state_info_vars = {
                k: tf.compat.v1.placeholder(tf.float32,
                                            shape=[None] * 2 + list(shape),
                                            name='infer_%s' % k)
                for k, shape in self._inference.state_info_specs
            }
            infer_state_info_vars_list = [
                infer_state_info_vars[k]
                for k in self._inference.state_info_keys
            ]

        extra_obs_var = [
            tf.cast(v, tf.float32) for v in policy_state_info_vars_list
        ]
        # pylint false alarm
        # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
        augmented_obs_var = tf.concat([obs_var] + extra_obs_var, axis=-1)
        extra_traj_var = [
            tf.cast(v, tf.float32) for v in infer_state_info_vars_list
        ]
        augmented_traj_var = tf.concat([trajectory_var] + extra_traj_var, -1)

        # Policy and encoder network loss and optimizer inputs
        policy_loss_inputs = graph_inputs(
            'PolicyLossInputs',
            augmented_obs_var=augmented_obs_var,
            augmented_traj_var=augmented_traj_var,
            task_var=task_var,
            action_var=action_var,
            reward_var=reward_var,
            baseline_var=baseline_var,
            valid_var=valid_var)
        policy_opt_inputs = graph_inputs(
            'PolicyOptInputs',
            obs_var=obs_var,
            action_var=action_var,
            reward_var=reward_var,
            baseline_var=baseline_var,
            trajectory_var=trajectory_var,
            task_var=task_var,
            latent_var=latent_var,
            valid_var=valid_var,
            policy_state_info_vars_list=policy_state_info_vars_list,
            embed_state_info_vars_list=embed_state_info_vars_list,
        )

        # Inference network loss and optimizer inputs
        inference_loss_inputs = graph_inputs('InferenceLossInputs',
                                             latent_var=latent_var,
                                             valid_var=valid_var)
        inference_opt_inputs = graph_inputs(
            'InferenceOptInputs',
            latent_var=latent_var,
            trajectory_var=trajectory_var,
            valid_var=valid_var,
            infer_state_info_vars_list=infer_state_info_vars_list,
        )

        return (policy_loss_inputs, policy_opt_inputs, inference_loss_inputs,
                inference_opt_inputs)

    def _build_policy_loss(self, i):
        """Build policy loss and other output tensors.

        Args:
            i (namedtuple): Collection of variables to compute policy loss.

        Returns:
            tf.Tensor: Policy loss.
            tf.Tensor: Mean policy KL divergence.

        """
        # pylint: disable=too-many-statements
        self.policy.build(i.augmented_obs_var, i.task_var)
        self._old_policy.build(i.augmented_obs_var, i.task_var)
        self._inference.build(i.augmented_traj_var)
        self._old_inference.build(i.augmented_traj_var)
        self.policy.model.parameters = self._old_policy.model.parameters
        self.policy.encoder.model.parameters = (
            self._old_policy.encoder.model.parameters)
        self._inference.model.parameters = self._old_inference.model.parameters

        pol_dist = self.policy.distribution
        old_pol_dist = self._old_policy.distribution

        # Entropy terms
        encoder_entropy, inference_ce, policy_entropy = (
            self._build_entropy_terms(i))

        rewards = i.reward_var

        # Augment the path rewards with entropy terms
        if self._maximum_entropy:
            with tf.name_scope('augmented_rewards'):
                rewards = (i.reward_var -
                           (self.inference_ce_coeff * inference_ce) +
                           (self._policy_ent_coeff * policy_entropy))

        with tf.name_scope('policy_loss'):
            with tf.name_scope('advantages'):
                adv = compute_advantages(self._discount,
                                         self._gae_lambda,
                                         self.max_path_length,
                                         i.baseline_var,
                                         rewards,
                                         name='advantages')
                adv = tf.reshape(adv, [-1, self.max_path_length])

            # Optionally normalize advantages
            eps = tf.constant(1e-8, dtype=tf.float32)
            if self._center_adv:
                adv = center_advs(adv, axes=[0], eps=eps)

            if self._positive_adv:
                adv = positive_advs(adv, eps)

            # Calculate loss function and KL divergence
            with tf.name_scope('kl'):
                kl = old_pol_dist.kl_divergence(pol_dist)
                pol_mean_kl = tf.reduce_mean(kl)

            # Calculate vanilla loss
            with tf.name_scope('vanilla_loss'):
                ll = pol_dist.log_prob(i.action_var, name='log_likelihood')
                vanilla = ll * adv

            # Calculate surrogate loss
            with tf.name_scope('surr_loss'):
                old_ll = old_pol_dist.log_prob(i.action_var)
                old_ll = tf.stop_gradient(old_ll)
                # Possibly overflow when ll-old_ll is large (e.g. >80)
                lr = tf.exp(ll - old_ll)

                surrogate = lr * adv

                surrogate = tf.debugging.check_numerics(surrogate,
                                                        message='surrogate')

            # Finalize objective function
            with tf.name_scope('loss'):
                if self._pg_loss == 'vanilla':
                    # VPG uses the vanilla objective
                    obj = tf.identity(vanilla, name='vanilla_obj')
                elif self._pg_loss == 'surrogate':
                    # TRPO uses the standard surrogate objective
                    obj = tf.identity(surrogate, name='surr_obj')
                elif self._pg_loss == 'surrogate_clip':
                    lr_clip = tf.clip_by_value(lr,
                                               1 - self._lr_clip_range,
                                               1 + self._lr_clip_range,
                                               name='lr_clip')
                    surr_clip = lr_clip * adv
                    obj = tf.minimum(surrogate, surr_clip, name='surr_obj')

                if self._entropy_regularzied:
                    obj += self._policy_ent_coeff * policy_entropy

                obj = tf.boolean_mask(obj, i.valid_var)
                # Maximize E[surrogate objective] by minimizing
                # -E_t[surrogate objective]
                loss = -tf.reduce_mean(obj)

                # Encoder entropy bonus
                loss -= self.encoder_ent_coeff * encoder_entropy

            encoder_mean_kl = self._build_encoder_kl(i)

            # Diagnostic functions
            self._f_policy_kl = tf.compat.v1.get_default_session(
            ).make_callable(pol_mean_kl,
                            feed_list=flatten_inputs(self._policy_opt_inputs))

            self._f_rewards = tf.compat.v1.get_default_session().make_callable(
                rewards, feed_list=flatten_inputs(self._policy_opt_inputs))

            returns = discounted_returns(self._discount,
                                         self.max_path_length,
                                         rewards,
                                         name='returns')
            self._f_returns = tf.compat.v1.get_default_session().make_callable(
                returns, feed_list=flatten_inputs(self._policy_opt_inputs))

        return loss, pol_mean_kl, encoder_mean_kl

    def _build_entropy_terms(self, i):
        """Build policy entropy tensor.

        Args:
            i (namedtuple): Collection of variables to compute policy loss.

        Returns:
            tf.Tensor: Policy entropy.

        """
        pol_dist = self.policy.distribution
        with tf.name_scope('entropy_terms'):
            # 1. Encoder distribution total entropy
            with tf.name_scope('encoder_entropy'):
                task_dim = self.policy.task_space.flat_dim
                # pylint false alarm
                # pylint: disable=no-value-for-parameter
                all_task_one_hots = tf.one_hot(np.arange(task_dim),
                                               task_dim,
                                               name='all_task_one_hots')
                encoder_dist_all_task, _, _ = self.policy.encoder.model.build(
                    all_task_one_hots, name='encoder_all_task')

                encoder_all_task_entropies = encoder_dist_all_task.entropy(
                    name='encoder_all_task_entropies')

                if self._use_softplus_entropy:
                    encoder_entropy = tf.nn.softplus(
                        encoder_all_task_entropies)

                encoder_entropy = tf.reduce_mean(encoder_entropy,
                                                 name='encoder_entropy')

            # 2. Infernece distribution cross-entropy (log-likelihood)
            with tf.name_scope('inference_ce'):
                # Build inference with trajectory windows

                traj_ll = self._inference.distribution.log_prob(
                    self.policy.encoder.distribution.sample(), name='traj_ll')

                inference_ce_raw = -traj_ll
                inference_ce = tf.clip_by_value(inference_ce_raw, -3, 3)

                if self._use_softplus_entropy:
                    inference_ce = tf.nn.softplus(inference_ce)

                if self._stop_ce_gradient:
                    inference_ce = tf.stop_gradient(inference_ce)

            # 3. Policy path entropies
            with tf.name_scope('policy_entropy'):
                if self._use_neg_logli_entropy:
                    policy_entropy = -pol_dist.log_prob(
                        i.action_var, name='policy_log_likeli')
                else:
                    policy_entropy = pol_dist.entropy()

                # This prevents entropy from becoming negative
                # for small policy std
                if self._use_softplus_entropy:
                    policy_entropy = tf.nn.softplus(policy_entropy)

                if self._stop_entropy_gradient:
                    policy_entropy = tf.stop_gradient(policy_entropy)

        # Diagnostic functions
        self._f_task_entropies = compile_function(flatten_inputs(
            self._policy_opt_inputs),
                                                  encoder_all_task_entropies,
                                                  log_name='f_task_entropies')
        self._f_encoder_entropy = compile_function(
            flatten_inputs(self._policy_opt_inputs),
            encoder_entropy,
            log_name='f_encoder_entropy')
        self._f_inference_ce = compile_function(
            flatten_inputs(self._policy_opt_inputs),
            tf.reduce_mean(inference_ce * i.valid_var),
            log_name='f_inference_ce')
        self._f_policy_entropy = compile_function(flatten_inputs(
            self._policy_opt_inputs),
                                                  policy_entropy,
                                                  log_name='f_policy_entropy')

        return encoder_entropy, inference_ce, policy_entropy

    def _build_encoder_kl(self, i):
        """Build graph for encoder KL divergence.

        Args:
            i (namedtuple): Collection of variables to compute encoder KL
                divergence.

        Returns:
            tf.Tensor: Encoder KL divergence.

        """
        del i

        dist = self.policy.encoder.distribution
        old_dist = self._old_policy.encoder.distribution

        with tf.name_scope('encoder_kl'):
            kl = old_dist.kl_divergence(dist)
            mean_kl = tf.reduce_mean(kl)

            # Diagnostic function
            self._f_encoder_kl = compile_function(flatten_inputs(
                self._policy_opt_inputs),
                                                  mean_kl,
                                                  log_name='f_encoder_kl')

            return mean_kl

    def _build_inference_loss(self, i):
        """Build loss function for the inference network.

        Args:
            i (namedtuple): Collection of variables to compute inference loss.

        Returns:
            tf.Tensor: Inference loss.

        """
        dist = self._inference.distribution
        old_dist = self._old_inference.distribution
        with tf.name_scope('infer_loss'):

            traj_ll = dist.log_prob(i.latent_var, name='traj_ll_2')

            # Calculate loss
            traj_gammas = tf.constant(float(self._discount),
                                      dtype=tf.float32,
                                      shape=[self.max_path_length])
            # pylint false alarm
            # pylint: disable=no-value-for-parameter
            traj_discounts = tf.compat.v1.cumprod(traj_gammas,
                                                  exclusive=True,
                                                  name='traj_discounts')
            discount_traj_ll = traj_discounts * traj_ll
            discount_traj_ll = tf.boolean_mask(discount_traj_ll, i.valid_var)

            with tf.name_scope('loss'):
                infer_loss = -tf.reduce_mean(discount_traj_ll,
                                             name='infer_loss')

            with tf.name_scope('kl'):
                # Calculate predicted encoder distributions for each timestep

                # Calculate KL divergence
                kl = old_dist.kl_divergence(dist)
                infer_kl = tf.reduce_mean(kl, name='infer_kl')

            return infer_loss, infer_kl

    def _policy_opt_input_values(self, samples_data):
        """Map rollout samples to the policy optimizer inputs.

        Args:
            samples_data (dict): Processed sample data.
                See process_samples() for details.

        Returns:
            list(np.ndarray): Flatten policy optimization input values.

        """
        policy_state_info_list = [
            samples_data['agent_infos'][k] for k in self.policy.state_info_keys
        ]
        embed_state_info_list = [
            samples_data['latent_infos'][k]
            for k in self.policy.encoder.state_info_keys
        ]
        policy_opt_input_values = self._policy_opt_inputs._replace(
            obs_var=samples_data['observations'],
            action_var=samples_data['actions'],
            reward_var=samples_data['rewards'],
            baseline_var=samples_data['baselines'],
            trajectory_var=samples_data['trajectories'],
            task_var=samples_data['tasks'],
            latent_var=samples_data['latents'],
            valid_var=samples_data['valids'],
            policy_state_info_vars_list=policy_state_info_list,
            embed_state_info_vars_list=embed_state_info_list,
        )

        return flatten_inputs(policy_opt_input_values)

    def _inference_opt_input_values(self, samples_data):
        """Map rollout samples to the inference optimizer inputs.

        Args:
            samples_data (dict): Processed sample data.
                See process_samples() for details.

        Returns:
            list(np.ndarray): Flatten inference optimization input values.

        """
        infer_state_info_list = [
            samples_data['trajectory_infos'][k]
            for k in self._inference.state_info_keys
        ]
        # pylint: disable=unexpected-keyword-arg
        inference_opt_input_values = self._inference_opt_inputs._replace(
            latent_var=samples_data['latents'],
            trajectory_var=samples_data['trajectories'],
            valid_var=samples_data['valids'],
            infer_state_info_vars_list=infer_state_info_list,
        )

        return flatten_inputs(inference_opt_input_values)

    def evaluate(self, policy_opt_input_values, samples_data):
        """Evaluate rewards and everything else.

        Args:
            policy_opt_input_values (list[np.ndarray]): Flattened
                policy optimization input values.
            samples_data (dict): Processed sample data.
                See process_samples() for details.

        Returns:
            dict: Processed sample data.

        """
        # pylint: disable=too-many-statements
        # Augment reward from baselines
        rewards_tensor = self._f_rewards(*policy_opt_input_values)
        returns_tensor = self._f_returns(*policy_opt_input_values)
        returns_tensor = np.squeeze(returns_tensor, -1)

        paths = samples_data['paths']
        valids = samples_data['valids']
        baselines = [path['baselines'] for path in paths]
        env_rewards = [path['rewards'] for path in paths]
        env_rewards = concat_tensor_list(env_rewards.copy())
        env_returns = [path['returns'] for path in paths]
        env_returns = concat_tensor_list(env_returns.copy())
        env_average_discounted_return = (np.mean(
            [path['returns'][0] for path in paths]))

        # Recompute parts of samples_data
        aug_rewards = []
        aug_returns = []
        for rew, ret, val, path in zip(rewards_tensor, returns_tensor, valids,
                                       paths):
            path['rewards'] = rew[val.astype(np.bool)]
            path['returns'] = ret[val.astype(np.bool)]
            aug_rewards.append(path['rewards'])
            aug_returns.append(path['returns'])
        aug_rewards = concat_tensor_list(aug_rewards)
        aug_returns = concat_tensor_list(aug_returns)
        samples_data['rewards'] = aug_rewards
        samples_data['returns'] = aug_returns

        # Calculate effect of the entropy terms
        d_rewards = np.mean(aug_rewards - env_rewards)
        tabular.record('{}/EntRewards'.format(self.policy.name), d_rewards)

        aug_average_discounted_return = (np.mean(
            [path['returns'][0] for path in paths]))
        d_returns = np.mean(aug_average_discounted_return -
                            env_average_discounted_return)
        tabular.record('{}/EntReturns'.format(self.policy.name), d_returns)

        # Calculate explained variance
        ev = np_tensor_utils.explained_variance_1d(np.concatenate(baselines),
                                                   aug_returns)
        tabular.record('{}/ExplainedVariance'.format(self._baseline.name), ev)

        inference_rmse = (samples_data['trajectory_infos']['mean'] -
                          samples_data['latents'])**2.
        inference_rmse = np.sqrt(inference_rmse.mean())
        tabular.record('Inference/RMSE', inference_rmse)

        inference_rrse = np_tensor_utils.rrse(
            samples_data['latents'], samples_data['trajectory_infos']['mean'])
        tabular.record('Inference/RRSE', inference_rrse)

        embed_ent = self._f_encoder_entropy(*policy_opt_input_values)
        tabular.record('{}/Encoder/Entropy'.format(self.policy.name),
                       embed_ent)

        infer_ce = self._f_inference_ce(*policy_opt_input_values)
        tabular.record('Inference/CrossEntropy', infer_ce)

        pol_ent = self._f_policy_entropy(*policy_opt_input_values)
        pol_ent = np.sum(pol_ent) / np.sum(samples_data['valids'])
        tabular.record('{}/Entropy'.format(self.policy.name), pol_ent)

        task_ents = self._f_task_entropies(*policy_opt_input_values)
        tasks = samples_data['tasks'][:, 0, :]
        _, task_indices = np.nonzero(tasks)
        path_lengths = np.sum(samples_data['valids'], axis=1)
        for t in range(self.policy.task_space.flat_dim):
            lengths = path_lengths[task_indices == t]
            completed = lengths < self.max_path_length
            pct_completed = np.mean(completed)
            tabular.record('Tasks/EpisodeLength/t={}'.format(t),
                           np.mean(lengths))
            tabular.record('Tasks/CompletionRate/t={}'.format(t),
                           pct_completed)
            tabular.record('Tasks/Entropy/t={}'.format(t), task_ents[t])

        return samples_data

    def visualize_distribution(self):
        """Visualize encoder distribution."""
        num_tasks = self.policy.task_space.flat_dim
        all_tasks = np.eye(num_tasks, num_tasks)
        _, latent_infos = self.policy.encoder.get_latents(all_tasks)

        for task in range(num_tasks):
            for i in range(self.policy.latent_space.flat_dim):
                # pylint: disable=protected-access
                stds = latent_infos['log_std'][task, i]

                norm = scipy.stats.norm(loc=latent_infos['mean'][task, i],
                                        scale=stds)
                samples = norm.rvs(100)
                hist = Histogram(samples)
                tabular.record('Encoder/task={},i={}'.format(task, i), hist)

    def _train_policy_and_encoder_networks(self, policy_opt_input_values):
        """Joint optimization of policy and encoder networks.

        Args:
            policy_opt_input_values (list(np.ndarray)): Flatten policy
                optimization input values.

        Returns:
            float: Policy loss after optimization.

        """
        logger.log('Computing loss before')
        loss_before = self._optimizer.loss(policy_opt_input_values)

        logger.log('Computing KL before')
        policy_kl_before = self._f_policy_kl(*policy_opt_input_values)
        embed_kl_before = self._f_encoder_kl(*policy_opt_input_values)

        logger.log('Optimizing')
        self._optimizer.optimize(policy_opt_input_values)

        logger.log('Computing KL after')
        policy_kl = self._f_policy_kl(*policy_opt_input_values)
        embed_kl = self._f_encoder_kl(*policy_opt_input_values)

        logger.log('Computing loss after')
        loss_after = self._optimizer.loss(policy_opt_input_values)
        tabular.record('{}/LossBefore'.format(self.policy.name), loss_before)
        tabular.record('{}/LossAfter'.format(self.policy.name), loss_after)
        tabular.record('{}/dLoss'.format(self.policy.name),
                       loss_before - loss_after)
        tabular.record('{}/KLBefore'.format(self.policy.name),
                       policy_kl_before)
        tabular.record('{}/KL'.format(self.policy.name), policy_kl)
        tabular.record('{}/Encoder/KLBefore'.format(self.policy.name),
                       embed_kl_before)
        tabular.record('{}/Encoder/KL'.format(self.policy.name), embed_kl)

        return loss_after

    def _train_inference_network(self, inference_opt_input_values):
        """Optimize inference network.

        Args:
            inference_opt_input_values (list(np.ndarray)): Flatten inference
                optimization input values.

        Returns:
            float: Inference loss after optmization.

        """
        logger.log('Optimizing inference network...')
        infer_loss_before = self.inference_optimizer.loss(
            inference_opt_input_values)
        tabular.record('Inference/Loss', infer_loss_before)
        self.inference_optimizer.optimize(inference_opt_input_values)
        infer_loss_after = self.inference_optimizer.loss(
            inference_opt_input_values)
        tabular.record('Inference/dLoss', infer_loss_before - infer_loss_after)

        return infer_loss_after

    @classmethod
    def _get_latent_space(cls, latent_dim):
        """Get latent space given latent length.

        Args:
            latent_dim (int): Length of latent.

        Returns:
            akro.Space: Space of latent.

        """
        latent_lb = np.zeros(latent_dim, )
        latent_up = np.ones(latent_dim, )
        return akro.Box(latent_lb, latent_up)

    @classmethod
    def get_encoder_spec(cls, task_space, latent_dim):
        """Get the embedding spec of the encoder.

        Args:
            task_space (akro.Space): Task spec.
            latent_dim (int): Latent dimension.

        Returns:
            metarl.InOutSpec: Encoder spec.

        """
        latent_space = cls._get_latent_space(latent_dim)
        return InOutSpec(task_space, latent_space)

    @classmethod
    def get_infer_spec(cls, env_spec, latent_dim, inference_window_size):
        """Get the embedding spec of the inference.

        Every `inference_window_size` timesteps in the trajectory will be used
        as the inference network input.

        Args:
            env_spec (metarl.envs.EnvSpec): Environment spec.
            latent_dim (int): Latent dimension.
            inference_window_size (int): Length of inference window.

        Returns:
            metarl.InOutSpec: Inference spec.

        """
        latent_space = cls._get_latent_space(latent_dim)

        obs_lb, obs_ub = env_spec.observation_space.bounds
        obs_lb_flat = env_spec.observation_space.flatten(obs_lb)
        obs_ub_flat = env_spec.observation_space.flatten(obs_ub)
        traj_lb = np.stack([obs_lb_flat] * inference_window_size)
        traj_ub = np.stack([obs_ub_flat] * inference_window_size)
        traj_space = akro.Box(traj_lb, traj_ub)

        return InOutSpec(traj_space, latent_space)

    def _check_entropy_configuration(self, entropy_method, center_adv,
                                     stop_entropy_gradient,
                                     use_neg_logli_entropy, policy_ent_coeff):
        """Check entropy configuration.

        Args:
            entropy_method (str): A string from: 'max', 'regularized',
                'no_entropy'. The type of entropy method to use. 'max' adds the
                dense entropy to the reward for each time step. 'regularized'
                adds the mean entropy to the surrogate objective. See
                https://arxiv.org/abs/1805.00909 for more details.
            center_adv (bool): Whether to rescale the advantages
                so that they have mean 0 and standard deviation 1.
            stop_entropy_gradient (bool): Whether to stop the entropy gradient.
            use_neg_logli_entropy (bool): Whether to estimate the entropy as
                the negative log likelihood of the action.
            policy_ent_coeff (float): The coefficient of the policy entropy.
                Setting it to zero would mean no entropy regularization.

        Raises:
            ValueError: If stop_gradient is False when entropy_method is max.
            ValueError: If policy_ent_coeff is non-zero when there is
                no entropy method.
            ValueError: If entropy_method is not one of 'max', 'regularized',
                'no_entropy'.

        """
        del use_neg_logli_entropy, center_adv

        if entropy_method == 'max':
            if not stop_entropy_gradient:
                raise ValueError('stop_gradient should be True when '
                                 'entropy_method is max')
            self._maximum_entropy = True
            self._entropy_regularzied = False
        elif entropy_method == 'regularized':
            self._maximum_entropy = False
            self._entropy_regularzied = True
        elif entropy_method == 'no_entropy':
            if policy_ent_coeff != 0.0:
                raise ValueError('policy_ent_coeff should be zero '
                                 'when there is no entropy method')
            self._maximum_entropy = False
            self._entropy_regularzied = False
        else:
            raise ValueError('Invalid entropy_method')

    def __getstate__(self):
        """Parameters to save in snapshot.

        Returns:
            dict: Parameters to save.

        """
        data = self.__dict__.copy()
        del data['_name_scope']
        del data['_inference_opt_inputs']
        del data['_policy_opt_inputs']
        del data['_f_inference_ce']
        del data['_f_task_entropies']
        del data['_f_encoder_entropy']
        del data['_f_encoder_kl']
        del data['_f_policy_entropy']
        del data['_f_policy_kl']
        del data['_f_rewards']
        del data['_f_returns']
        return data

    def __setstate__(self, state):
        """Parameters to restore from snapshot.

        Args:
            state (dict): Parameters to restore from.

        """
        self.__dict__ = state
        self._name_scope = tf.name_scope(self._name)
        self.init_opt()
