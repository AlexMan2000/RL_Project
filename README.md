# How to start
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
| Model-based (`--method model_based`) | NA | MDP Planning |

### Example Commands
```bash
# For policy-based algorithms
python main.py --num-episodes 10 --no-save --method policy_based --policy_based_model pgmc

# For value-based algorithms
python main.py --num-episodes 10 --no-save --method value_based --value_based_model mlp

# For model-based algorithms
python main.py --num-episodes 100 --no-save --method model_based
```


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

## Available Tools

### 1. Evaluation Script (`evaluate.py`)

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

### 2. Visualization Tool (`visualize.py`)

Interactive visualization of trained agents:
```bash
python experiments/visualize.py
```

Features:
- Real-time game board visualization
- Action probability display (for applicable methods)
- Step-by-step gameplay observation
- Multiple episode playthrough

## Running Experiments

1. **Quick Evaluation**
```bash
# Run with default settings
python experiments/evaluate.py

# Customize number of episodes and runs
python experiments/evaluate.py --num-episodes 2000 --num-runs 10
```

2. **Visualize Trained Agent**
```bash
# Interactive visualization
python experiments/visualize.py

# Adjust visualization speed
python experiments/visualize.py --delay 0.3
```

3. **Analyzing Results**
- Results are saved in `experiment_results/` with timestamp
- Each run creates:
  - `learning_curves.png`: Visual comparison
  - `comparison_table.csv`: Detailed metrics
  - `raw_results.json`: Complete data

## Interpreting Results

1. **Learning Curves**
   - Faster convergence = Better sample efficiency
   - Higher final value = Better performance
   - Smaller variance = More stable learning

2. **Computational Metrics**
   - Training time
   - Memory usage
   - Number of model updates

3. **Performance Comparison**
   - Success rate in reaching 2048
   - Average and best scores
   - Policy stability
   - Sample efficiency 