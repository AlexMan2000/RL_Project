### How to start
```
conda env create -f environment.yml --prefix ./venv
conda activate ./venv
```

### Available Models

| Method | Model Option | Description |
|--------|-------------|-------------|
| Value-based (`--method value_based`) | `--value_based_model cnn` | CNN-based DQN |
| | `--value_based_model mlp` | MLP-based DQN |
| Policy-based (`--method policy_based`) | `--policy_based_model pgmc` | Policy Gradient Monte Carlo |
| | `--policy_based_model actor_critic` | Actor-Critic |
| | `--policy_based_model trpo` | Trust Region Policy Optimization |
| | `--policy_based_model ppo` | Proximal Policy Optimization |

### Example Commands
```bash
# For policy-based algorithms
python main.py --num-episodes 10 --no-save --method policy_based --policy_based_model pgmc

# For value-based algorithms
python main.py --num-episodes 10 --no-save --method value_based --value_based_model mlp

# For model-based algorithms
python main.py --num-episodes 100 --no-save --method model_based
```
