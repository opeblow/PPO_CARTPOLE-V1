"""Tests for PPO network."""

import torch

from ppo.network import ActorCriticNetwork


class TestActorCriticNetwork:
    """Tests for ActorCriticNetwork class."""

    def test_initialization(self):
        """Test network initialization."""
        state_dim = 4
        action_dim = 2
        hidden_size = 64

        network = ActorCriticNetwork(state_dim, action_dim, hidden_size)

        assert network._state_dim == state_dim
        assert network._action_dim == action_dim

    def test_actor_output_shape(self):
        """Test actor network output shape."""
        state_dim = 4
        action_dim = 2
        batch_size = 32

        network = ActorCriticNetwork(state_dim, action_dim)
        state = torch.randn(batch_size, state_dim)

        probs = network.actor(state)
        assert probs.shape == (batch_size, action_dim)

    def test_critic_output_shape(self):
        """Test critic network output shape."""
        state_dim = 4
        batch_size = 32

        network = ActorCriticNetwork(state_dim, action_dim=2)
        state = torch.randn(batch_size, state_dim)

        values = network.critic(state)
        assert values.shape == (batch_size, 1)

    def test_act_single_state(self):
        """Test act method with single state."""
        network = ActorCriticNetwork(state_dim=4, action_dim=2)
        state = torch.randn(1, 4)

        action, log_prob, entropy = network.act(state)

        assert action.shape == (1,)
        assert log_prob.shape == (1,)
        assert entropy.shape == (1,)

    def test_act_probs_sum_to_one(self):
        """Test that action probabilities sum to 1."""
        network = ActorCriticNetwork(state_dim=4, action_dim=3)
        state = torch.randn(1, 4)

        probs = network.actor(state)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-5)

    def test_evaluate_batch(self):
        """Test evaluate method with batch."""
        batch_size = 16
        network = ActorCriticNetwork(state_dim=4, action_dim=2)

        state = torch.randn(batch_size, 4)
        action = torch.randint(0, 2, (batch_size,))

        log_probs, values, entropy = network.evaluate(state, action)

        assert log_probs.shape == (batch_size,)
        assert values.shape == (batch_size,)
        assert entropy.shape == (batch_size,)

    def test_forward_raises_error(self):
        """Test that forward raises NotImplementedError."""
        network = ActorCriticNetwork(state_dim=4, action_dim=2)

        try:
            network.forward()
            assert False, "Should have raised NotImplementedError"
        except NotImplementedError:
            pass
