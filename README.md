# Lesion Detection Using Value-Based Reinforcement Learning (DQN & DDQN)

This project implements a value-based reinforcement learning approach using Deep Q-Networks (DQN) and Double DQN (DDQN) for detecting lesions in segmented brain images. The agent navigates a grid to accurately identify and localize lesions in MRI scans.

## Project Structure

The project is organized into three main folders:

1. **DDQN Model**:
   - Implements the Double DQN approach for more stable training and better lesion detection.
   - **Key File**: `main.py`
   - **Run Command**: 
     ```bash
     python main.py
     ```

2. **DQN Model**:
   - Implements the standard DQN algorithm, where the agent uses a single model for both policy and evaluation.
   - **Key File**: `main.py`
   - **Run Command**: 
     ```bash
     python main.py
     ```

3. **Prototype**:
   - Contains the initial prototype of the lesion detection system. This can be used for testing and quick iterations.

## How to Run the Project

### Requirements
Before running the project, ensure the following dependencies are installed:

```bash
pip install pygame torch matplotlib numpy opencv-python
```

### Running DDQN or DQN Models

Both models are set up to train an agent that navigates through segmented MRI images. To run either model, use the following commands:

#### For DDQN:
```bash
cd DDQN_Model
python main.py
```

#### For DQN:
```bash
cd DQN_Model
python main.py
```

### Command-Line Usage
Each model is executed through `main.py`, which trains the agent and logs rewards and performance. Example usage:

```bash
python main.py
```

This command will:
1. Train the agent on the MRI images.
2. Save images labeled with detected lesions.
3. Output episode durations and rewards.

## Key Files Explanation

- **`main.py`**: Main training script that initializes the environment and agent, runs the training loop, and logs results.
- **`environment.py`**: Defines the environment for the agent, where MRI images are loaded and the grid navigation is implemented.
- **`agent.py`**: Contains the logic for the DDQN agent, including action selection and training of the neural network.
- **`model.py`**: Defines the architecture of the Deep Q-Network (DQN) used by the agent.
- **`replay_memory.py`**: Implements experience replay, which stores past experiences and samples batches for training.
- **`test.py`**: Script to evaluate the trained model on new images and visualize the agent’s performance.

## Results

During training, the agent navigates a grid over an MRI image. The model outputs include:
- **Rewards**: Total rewards per episode.
- **Losses**: Average loss for each training episode.
- **Labeled Images**: Images saved with detected lesion regions highlighted.

### Plotting Results
The training script automatically generates plots:
- **Real-Time Rewards**: Shows how rewards evolve over training episodes.
- **Cumulative Time**: Tracks the cumulative training time for each milestone episode.

## License

This project is licensed under the **Apache License, Version 2.0**. See the [LICENSE](LICENSE) file for details.

```text
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

