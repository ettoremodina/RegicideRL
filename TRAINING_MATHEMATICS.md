# Mathematical Foundation of Regicide RL Training

## Overview

This document provides a rigorous mathematical explanation of the REINFORCE algorithm implementation used in the Regicide card game training system. The implementation combines policy gradient methods with card-aware neural architectures to learn optimal gameplay strategies.

### Extensions and Improvements

Two key improvements can be made to the current REINFORCE implementation:

1. **Replay Buffer Integration**: Transition to actor-critic methods for sample efficiency
2. **Exploration-Exploitation Control**: Add explicit mechanisms to balance exploration vs exploitation

## 1. Markov Decision Process Formulation

### 1.1 State Space
The Regicide environment is modeled as a Markov Decision Process (MDP) with tuple $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$ where:

- **State Space** $\mathcal{S}$: Card-aware structured observations containing:
  ```python
  # From card_aware_policy.py line 100-120
  observation = {
      'hand_cards': torch.Tensor,      # Player's hand cards
      'game_state': torch.Tensor,      # Game phase, bosses killed, etc.
      'discard_pile_cards': torch.Tensor,  # Cards in discard pile
      'enemy_card': torch.Tensor,      # Current boss/enemy
      'hand_size': torch.Tensor,       # Number of cards in hand
      'num_valid_actions': int         # Number of legal actions
  }
  ```

- **Action Space** $\mathcal{A}$: Variable-sized discrete action space where valid actions depend on current game state and hand composition

- **Transition Probabilities** $\mathcal{P}(s'|s,a)$: Deterministic transitions governed by Regicide game rules

- **Reward Function** $\mathcal{R}(s,a,s')$: Sparse rewards based on boss defeats and game outcomes

- **Discount Factor** $\gamma = 0.95$: Chosen to prioritize immediate progress while maintaining long-term planning

### 1.2 Policy Parameterization

The policy $\pi_\theta(a|s)$ is parameterized by a card-aware neural network with parameters $\theta$. The policy outputs action probabilities:

$$\pi_\theta(a|s) = \frac{\exp(f_\theta(s,a))}{\sum_{a' \in \mathcal{A}(s)} \exp(f_\theta(s,a'))}$$

where $f_\theta(s,a)$ is the neural network's action scoring function and $\mathcal{A}(s)$ represents valid actions in state $s$.

## 2. Neural Network Architecture

### 2.1 Card Embeddings

Each card $c \in \{1,2,...,54\}$ is mapped to a dense representation:

$$\mathbf{e}_c = \text{Embed}(c) \in \mathbb{R}^{d_{embed}}$$

where $d_{embed} = 12$ in our implementation.

```python
# From card_aware_policy.py line 31-32
self.card_embedding = nn.Embedding(54, card_embed_dim, padding_idx=0)
self.enemy_embedding = nn.Embedding(54, card_embed_dim, padding_idx=0)
```

### 2.2 Hand Representation

For a hand $H = \{c_1, c_2, ..., c_k\}$ with $k \leq$ `max_hand_size`, the hand context is computed as:

$$\mathbf{h} = \frac{1}{|H|} \sum_{i=1}^{|H|} \mathbf{e}_{c_i}$$

This averages card embeddings to create a fixed-size hand representation.

```python
# From card_aware_policy.py line 129-135
hand_lengths = torch.clamp(hand_lengths, min=1.0)
hand_context = attended_hand.sum(dim=1) / hand_lengths
```

### 2.3 Context Integration

The full context vector combines multiple information sources:

$$\mathbf{z} = [\mathbf{h}; \mathbf{e}_{enemy}; \phi(\mathbf{g}); \phi_{discard}(\mathbf{d})]$$

where:
- $\mathbf{h}$ is the hand context
- $\mathbf{e}_{enemy}$ is the current enemy/boss embedding
- $\phi(\mathbf{g})$ encodes game state features
- $\phi_{discard}(\mathbf{d})$ encodes discard pile information

```python
# From card_aware_policy.py line 141-144
combined_context = torch.cat([hand_context, enemy_embedding, game_context, discard_context], dim=-1)
context_features = self.context_encoder(combined_context)
```

### 2.4 Action Scoring

For each valid action $a$, the scoring function computes:

$$f_\theta(s,a) = \text{MLP}([\mathbf{z}; \mathbf{a}])$$

where $\mathbf{a}$ is the action's card representation (average of cards involved in the action).

```python
# From card_aware_policy.py line 147-170
action_features = torch.cat([context_features, action_card_repr], dim=-1)
score = self.action_scorer(action_features)
```

## 3. REINFORCE Algorithm

### 3.1 Policy Gradient Theorem

The REINFORCE algorithm optimizes the expected return $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$ where $\tau$ represents a trajectory.

The policy gradient is:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]$$

where $G_t$ is the return from time step $t$.

### 3.2 Return Computation

Returns are computed using the discounted sum of future rewards:

$$G_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k+1}$$

```python
# From streamlined_training.py line 148-154
def compute_returns(self, rewards: List[float]) -> List[float]:
    returns = []
    G = 0
    
    for reward in reversed(rewards):
        G = reward + self.gamma * G
        returns.insert(0, G)
    
    return returns
```

### 3.3 Baseline Subtraction

To reduce variance, we subtract a baseline $b$ from the returns:

$$\text{Advantage}_t = G_t - b$$

where $b = \frac{1}{T}\sum_{t=0}^{T-1} G_t$ is the mean return for the episode.

```python
# From streamlined_training.py line 178-183
if len(returns) > 1:
    baseline = returns.mean()
    advantages = returns - baseline
else:
    advantages = returns
```

### 3.4 Policy Loss

The policy loss (negative log-likelihood weighted by advantages) is:

$$L(\theta) = -\frac{1}{T}\sum_{t=0}^{T-1} \log \pi_\theta(a_t|s_t) \cdot \text{Advantage}_t$$

```python
# From streamlined_training.py line 185
policy_loss = -(log_probs * advantages).mean()
```

### 3.5 Gradient Update

Parameters are updated using AdamW optimizer with gradient clipping:

$$\theta \leftarrow \theta - \alpha \cdot \text{clip}(\nabla_\theta L(\theta), \|\cdot\|_2 \leq 0.5)$$

```python
# From streamlined_training.py line 187-193
self.optimizer.zero_grad()
policy_loss.backward()
torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
self.optimizer.step()
```

## 4. Optimization Details

### 4.1 AdamW Optimizer

The AdamW optimizer implements adaptive learning rates with weight decay:

$$\theta_{t+1} = \theta_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \lambda \theta_t$$

where:
- $\alpha = 0.0015$ (learning rate)
- $\lambda = 10^{-4}$ (weight decay)
- $\epsilon = 10^{-8}$ (numerical stability)

```python
# From streamlined_training.py line 52-57
self.optimizer = optim.AdamW(
    self.policy.parameters(), 
    lr=learning_rate,
    weight_decay=1e-4,
    eps=1e-8
)
```

### 4.2 Learning Rate Scheduling

A plateau scheduler reduces learning rate when performance stagnates:

$$\alpha_{new} = \alpha_{old} \cdot 0.8 \text{ if no improvement for 1000 episodes}$$

```python
# From streamlined_training.py line 59-65
self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    self.optimizer, 
    mode='max',
    factor=0.8,
    patience=1000,
    min_lr=1e-5
)
```

### 4.3 Gradient Clipping

Gradient clipping prevents exploding gradients by constraining the L2 norm:

$$\nabla_\theta L(\theta) \leftarrow \begin{cases}
\nabla_\theta L(\theta) & \text{if } \|\nabla_\theta L(\theta)\|_2 \leq 0.5 \\
0.5 \cdot \frac{\nabla_\theta L(\theta)}{\|\nabla_\theta L(\theta)\|_2} & \text{otherwise}
\end{cases}$$

## 5. Training Procedure

### 5.1 Episode Collection

Each training iteration follows this sequence:

1. **Reset Environment**: Initialize new game state $s_0$
2. **Action Selection**: Sample action $a_t \sim \pi_\theta(\cdot|s_t)$
3. **Environment Step**: Receive $(s_{t+1}, r_t, \text{done})$
4. **Store Trajectory**: Save $(s_t, a_t, \log \pi_\theta(a_t|s_t), r_t)$
5. **Repeat**: Until episode termination

```python
# From streamlined_training.py line 72-110
def collect_episode(self, render: bool = False):
    observations, actions, log_probs, rewards = [], [], [], []
    obs, info = self.env.reset()
    
    while True:
        action, log_prob = self.policy.get_action(obs)
        next_obs, reward, terminated, truncated, next_info = self.env.step(action)
        
        observations.append(obs)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        
        if terminated or truncated:
            break
        obs = next_obs
```

### 5.2 Policy Update

After each episode:

1. **Compute Returns**: Calculate $G_t$ for all time steps
2. **Compute Advantages**: Subtract baseline from returns
3. **Compute Loss**: Calculate policy gradient loss
4. **Update Parameters**: Apply gradients with clipping

```python
# From streamlined_training.py line 158-203
def train_episode(self, render: bool = False):
    # Collect episode data
    observations, actions, log_probs, rewards, ... = self.collect_episode(render)
    
    # Compute returns and advantages
    returns = self.compute_returns(rewards)
    advantages = returns - returns.mean()
    
    # Policy gradient update
    policy_loss = -(log_probs * advantages).mean()
    self.optimizer.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
    self.optimizer.step()
```

## 6. Convergence Properties

### 6.1 Theoretical Guarantees

REINFORCE with baseline has the following convergence properties:

1. **Unbiased Gradient Estimates**: $\mathbb{E}[\nabla_\theta L(\theta)] = \nabla_\theta J(\theta)$
2. **Variance Reduction**: Baseline subtraction reduces gradient variance without introducing bias
3. **Convergence**: Under suitable conditions (bounded rewards, Lipschitz policy), converges to local optimum

### 6.2 Practical Considerations

The implementation includes several techniques to improve convergence:

1. **Xavier Initialization**: Proper weight initialization for stable training
2. **Dropout Regularization**: Prevents overfitting (rate = 0.1)
3. **Weight Decay**: L2 regularization in optimizer
4. **Gradient Clipping**: Prevents exploding gradients
5. **Learning Rate Scheduling**: Adaptive learning rate reduction

```python
# From card_aware_policy.py line 68-77
def _init_weights(self, m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=0.8)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
```

## 7. Implementation Hyperparameters

| Parameter | Value | Mathematical Role |
|-----------|-------|-------------------|
| Learning Rate ($\alpha$) | 0.0015 | Step size in gradient descent |
| Discount Factor ($\gamma$) | 0.95 | Future reward weighting |
| Card Embedding Dim | 12 | Dimensionality of card representations |
| Hidden Dimension | 32 | Network capacity |
| Gradient Clip Norm | 0.5 | Maximum gradient magnitude |
| Weight Decay ($\lambda$) | $10^{-4}$ | L2 regularization strength |
| Episodes | 50,000 | Total training iterations |

## 8. Mathematical Intuition

### 8.1 Why REINFORCE Works

REINFORCE optimizes the policy by:

1. **Increasing Probability** of actions that led to high returns
2. **Decreasing Probability** of actions that led to low returns
3. **Proportional Updates** based on advantage magnitude

The log-probability gradient has the intuitive form:

$$\nabla_\theta \log \pi_\theta(a|s) = \frac{\nabla_\theta \pi_\theta(a|s)}{\pi_\theta(a|s)}$$

This creates larger updates for less probable actions that turned out well.

### 8.2 Card-Aware Architecture Benefits

The card embedding approach provides:

1. **Generalization**: Similar cards have similar representations
2. **Compositional Understanding**: Action scoring considers card combinations
3. **Scalability**: Fixed-size representations regardless of hand size
4. **Interpretability**: Embeddings capture card relationships

### 8.3 Variance Reduction Techniques

The implementation uses multiple variance reduction methods:

1. **Baseline Subtraction**: Reduces variance without bias
2. **Gradient Clipping**: Prevents extreme updates
3. **Batch Normalization Effects**: Through proper initialization
4. **Regularization**: Prevents overfitting and improves generalization

## 9. Expected Learning Dynamics

The training process exhibits several phases:

1. **Exploration Phase** (Episodes 1-10,000): High variance, random-like behavior
2. **Learning Phase** (Episodes 10,000-30,000): Gradual improvement, pattern recognition
3. **Refinement Phase** (Episodes 30,000+): Fine-tuning strategies, convergence

The mathematical framework ensures that with sufficient exploration and proper hyperparameters, the policy will converge to a locally optimal strategy for the Regicide game.

## 10. Algorithmic Extensions

### 10.1 Replay Buffer Integration

#### Mathematical Foundation

The current REINFORCE algorithm is **on-policy**, meaning it only learns from trajectories generated by the current policy. A replay buffer enables **off-policy** learning, where we can reuse past experiences.

To integrate a replay buffer, we need to transition to an **Actor-Critic** architecture:

**Actor (Policy Network)**: $\pi_\theta(a|s)$ - same as current implementation
**Critic (Value Network)**: $V_\phi(s)$ - estimates state values

The actor-critic loss becomes:
$$L_{total} = L_{actor} + L_{critic}$$

where:
$$L_{actor} = -\mathbb{E}[(r + \gamma V_\phi(s') - V_\phi(s)) \log \pi_\theta(a|s)]$$
$$L_{critic} = \mathbb{E}[(r + \gamma V_\phi(s') - V_\phi(s))^2]$$

#### Implementation Changes Required

```python
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

class ActorCriticPolicy(CardAwarePolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add critic network
        self.critic = nn.Sequential(
            nn.Linear(self.context_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
    
    def get_value(self, observation):
        """Estimate state value"""
        context = self.encode_context(observation)
        return self.critic(context)
```

#### Advantages of Replay Buffer

1. **Sample Efficiency**: Reuse past experiences, reducing environment interactions
2. **Stability**: Less correlated updates, more stable learning
3. **Data Efficiency**: Learn from good experiences multiple times

#### Mathematical Trade-offs

- **Bias-Variance**: Off-policy methods can introduce bias but reduce variance
- **Distribution Mismatch**: Old experiences may not reflect current policy distribution
- **Importance Sampling**: May need to weight samples by $\frac{\pi_\theta(a|s)}{\pi_{old}(a|s)}$

### 10.2 Exploration-Exploitation Control

#### Current Exploration Mechanism

Your current implementation uses **stochastic policy sampling**:

```python
# From card_aware_policy.py
action_dist = torch.distributions.Categorical(valid_probs)
action = action_dist.sample()  # Stochastic sampling
```

The exploration level is implicitly controlled by the **entropy** of the policy distribution:

$$H(\pi_\theta) = -\sum_{a} \pi_\theta(a|s) \log \pi_\theta(a|s)$$

#### Explicit Exploration Control Methods

**1. Entropy Regularization**

Add entropy bonus to encourage exploration:

$$L_{total} = L_{policy} - \beta H(\pi_\theta)$$

where $\beta > 0$ controls exploration strength.

```python
# Modified loss computation
entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
policy_loss = -(log_probs * advantages).mean() - beta * entropy
```

**2. Temperature Scaling**

Control exploration via temperature parameter $\tau$:

$$\pi_\tau(a|s) = \frac{\exp(f_\theta(s,a)/\tau)}{\sum_{a'} \exp(f_\theta(s,a')/\tau)}$$

- $\tau \to 0$: More deterministic (exploitation)
- $\tau \to \infty$: More uniform (exploration)

```python
class TemperatureControlledPolicy(CardAwarePolicy):
    def __init__(self, *args, initial_temp=1.0, temp_decay=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = initial_temp
        self.temp_decay = temp_decay
    
    def get_action(self, observation, action_mask=None):
        logits = self.forward(observation)
        # Apply temperature scaling
        scaled_logits = logits / self.temperature
        
        # Rest of sampling logic...
        probs = F.softmax(scaled_logits, dim=-1)
        action_dist = torch.distributions.Categorical(probs)
        return action_dist.sample(), action_dist.log_prob(action)
    
    def decay_temperature(self):
        self.temperature *= self.temp_decay
        self.temperature = max(self.temperature, 0.1)  # Minimum temperature
```

**3. Epsilon-Greedy with Neural Networks**

Combine deterministic policy with random exploration:

$$a = \begin{cases}
\arg\max_a \pi_\theta(a|s) & \text{with probability } 1-\epsilon \\
\text{random action} & \text{with probability } \epsilon
\end{cases}$$

```python
def epsilon_greedy_action(self, observation, epsilon=0.1):
    if random.random() < epsilon:
        # Random exploration
        num_valid = observation['num_valid_actions']
        return random.randint(0, num_valid - 1)
    else:
        # Greedy exploitation
        logits = self.forward(observation)
        return torch.argmax(logits[:observation['num_valid_actions']]).item()
```

**4. Curiosity-Driven Exploration**

Add intrinsic motivation based on prediction error:

$$r_{total} = r_{extrinsic} + \eta \cdot r_{intrinsic}$$

where $r_{intrinsic}$ measures novelty or prediction error.

#### Exploration Schedule Implementation

```python
class ExplorationScheduler:
    def __init__(self, method='entropy', initial_value=1.0, 
                 final_value=0.1, decay_episodes=30000):
        self.method = method
        self.initial = initial_value
        self.final = final_value
        self.decay_episodes = decay_episodes
    
    def get_exploration_param(self, episode):
        """Get current exploration parameter"""
        progress = min(episode / self.decay_episodes, 1.0)
        
        if self.method == 'linear':
            return self.initial + progress * (self.final - self.initial)
        elif self.method == 'exponential':
            return self.initial * (self.final / self.initial) ** progress
        elif self.method == 'cosine':
            return self.final + 0.5 * (self.initial - self.final) * \
                   (1 + np.cos(np.pi * progress))
```

#### Modified Training Loop

```python
def train_with_exploration_control(self, num_episodes, exploration_method='entropy'):
    scheduler = ExplorationScheduler(method=exploration_method)
    
    for episode in range(num_episodes):
        # Update exploration parameter
        if exploration_method == 'temperature':
            self.policy.temperature = scheduler.get_exploration_param(episode)
        elif exploration_method == 'epsilon':
            epsilon = scheduler.get_exploration_param(episode)
            # Use epsilon in action selection
        elif exploration_method == 'entropy':
            beta = scheduler.get_exploration_param(episode)
            # Use beta in loss computation
        
        # Standard training step with modified exploration
        results = self.train_episode_with_exploration(beta or epsilon)
```

#### Mathematical Intuition

**Exploration-Exploitation Trade-off**:

1. **Early Training**: High exploration to discover good strategies
2. **Mid Training**: Balanced exploration-exploitation 
3. **Late Training**: Low exploration to refine learned strategies

**Entropy vs Performance Relationship**:
- High entropy → High exploration → High variance → Slower convergence
- Low entropy → Low exploration → Risk of local optima → Faster convergence

#### Practical Recommendations

For your Regicide implementation:

1. **Start with entropy regularization** (easiest to implement)
2. **Use temperature decay** from 1.0 to 0.2 over 30,000 episodes
3. **Monitor policy entropy** during training to ensure sufficient exploration
4. **Consider replay buffer** if sample efficiency becomes important

The mathematical framework provides flexibility to experiment with different exploration strategies while maintaining the core REINFORCE algorithm structure.
