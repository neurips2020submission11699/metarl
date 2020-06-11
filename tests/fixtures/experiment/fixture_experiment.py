"""A dummy experiment fixture."""
from metarl.envs import MetaRLEnv
from metarl.experiment import LocalTFRunner
from metarl.np.baselines import LinearFeatureBaseline
from metarl.tf.algos import VPG
from metarl.tf.policies import CategoricalMLPPolicy


# pylint: disable=missing-return-type-doc
def fixture_exp(snapshot_config, sess):
    """Dummy fixture experiment function.

    Args:
        snapshot_config (metarl.experiment.SnapshotConfig): The snapshot
            configuration used by LocalRunner to create the snapshotter.
            If None, it will create one with default settings.
        sess (tf.Session): An optional TensorFlow session.
              A new session will be created immediately if not provided.

    Returns:
        np.ndarray: Values of the parameters evaluated in
            the current session

    """
    with LocalTFRunner(snapshot_config=snapshot_config, sess=sess) as runner:
        env = MetaRLEnv(env_name='CartPole-v1')

        policy = CategoricalMLPPolicy(name='policy',
                                      env_spec=env.spec,
                                      hidden_sizes=(8, 8))

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = VPG(env_spec=env.spec,
                   policy=policy,
                   baseline=baseline,
                   max_path_length=100,
                   discount=0.99,
                   optimizer_args=dict(learning_rate=0.01, ))

        runner.setup(algo, env)
        runner.train(n_epochs=5, batch_size=100)

        return policy.get_param_values()
