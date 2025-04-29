# How to start
```
conda env create -f environment.yml --prefix ./venv
conda activate ./venv
```

## Available Models

| Method | Model Option | Description |
|--------|-------------|-------------|
| Value-based (`--method value_based`) | `--value_based_model cnn` | CNN-based DQN |
| | `--value_based_model mlp` | MLP-based DQN |
| Policy-based (`--method policy_based`) | `--policy_based_model pgmc` | Policy Gradient Monte Carlo |
| | `--policy_based_model actor_critic` | Actor-Critic |
| | `--policy_based_model trpo` | Trust Region Policy Optimization |
| | `--policy_based_model ppo` | Proximal Policy Optimization |
| Model-based (`--method model_based`) | NA | MDP Planning |

### Example Commands
```bash
# For policy-based algorithms
python train.py --num-episodes 10 --no-save --method policy_based --policy_based_model pgmc

# For value-based algorithms
python train.py --num-episodes 10 --no-save --method value_based --value_based_model mlp

# For model-based algorithms
python train.py --num-episodes 100 --no-save --method model_based
```


# Training
## Train with default parameters
To train for a particular model: execute:
```bash
# Train DQN model for 2048, train for 100 episodes, and save model checkpoints every 10 episodes with visualization graph
python train.py --num-episodes 100 --log-every 10 --method value_based --value_based_model mlp

# Train DQN model for 2048, train for 100 episodes, and don't save model checkpoints, but with visualization graph
python train.py --num-episodes 100 --log-every 10 --method value_based --value_based_model mlp --no-save
```


## Adjust parameters 
### For value-based model (mlp, cnn)
You can adjust the parameter for value-based models in the config.py RLConfig class where you can specify the `batch_size`(for small number of episodes like 200, `batch_size` shouldn't be too big, 16 or 32 is fine), the `epsilon`, and `gamma`. Or you can modify the model structure in `value_based/mlp_model.py`(or `value_based/cnn_model.py`)


### For policy-based model(ppo, actor_critic, trpo, pgmc)
You can adjust the parameter for policy-based models in the correponding model files. For example for PPO we can adjust `clip_epsilon` ,`n_epochs`, `gae_lambda`, `value_clip_range`, `entropy_coef` and `max_grad_norm` in the `__init__` function of PPOAgent. While the actor-critic neural network model parameters like `hidden_size` should be adjusted in the `RL_Config.py`.
        

### For model-based models


# Experiments and Evaluation

This directory contains tools for evaluating and comparing different RL methods on the 2048 game environment.

## Evaluation Criteria

1. **Effectiveness**
   - Agent's ability to consistently reach the 2048 tile and beyond
   - Measured through success rate and highest tile achieved

2. **Computational Cost**
   - Number of training episodes required
   - Training time
   - Memory usage
   - Updates per episode

3. **Performance Metrics**
   - Average game score
   - Highest tile achieved
   - Policy stability across multiple runs
   - Learning curve characteristics

## Evaluation Script (`evaluate.py`)

Runs comprehensive experiments comparing different methods:
```bash
python experiments/evaluate.py
```

This will:
- Run each method multiple times
- Generate learning curves
- Create performance comparison tables
- Save results to `experiment_results/`

Output includes:
- Learning curves plot
- Training time comparison
- Best scores distribution
- Episode length analysis
- Detailed CSV report

## Running Experiments

1. **Quick Evaluation**
```bash
# Run with default settings, will compute the avg score with 100 generated episodes using the trained q-values
python evaluate.py --checkpoint path/to/checkpoint.pt --algorithm value_based --value_based_model mlp --num-episodes 100
```