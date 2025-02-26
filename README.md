# Detecting Antagonistic Agents in Robot Swarms Performing a Surveillance Task

Authors: Ingeborg Wenger

## Reference
A reference to the paper "_Discovering Antagonists in Networks of Systems:
Robot Deployment_" will be added once it is published.

## Abstract
A contextual anomaly detection method is proposed and applied to the phys-
ical motions of a robot swarm executing a coverage task. Using simulations
of a swarm’s normal behavior, a normalizing flow is trained to predict the
likelihood of a robot motion within the current context of its environment.
During application, the predicted likelihood of the observed motions is used
by a detection criterion that categorizes a robot agent as normal or antag-
onistic. The proposed method is evaluated on five different strategies of
antagonistic behavior. Importantly, only readily available simulated data
of normal robot behavior is used for training such that the nature of the
anomalies need not be known beforehand. The best detection criterion cor-
rectly categorizes at least 80% of each antagonistic type while maintaining
a false positive rate of less than 5% for normal robot agents. Additionally,
the method is validated in hardware experiments, yielding results similar to
the simulated scenarios. Compared to the state-of-the-art approach, both
the predictive performance of the normalizing flow and the robustness of the
detection criterion are increased.

## Videos

https://github.com/user-attachments/assets/96813752-230e-4bc1-bd13-92efd88f5c4e

https://github.com/user-attachments/assets/88e0cbdf-845e-4afd-bf05-c52e88973685

## Contents of the Repository
- `hardware_experiments`: Data collected in hardware experiments using a swarm of HERA robots.
- `lcm_types`: Message types used for communication within the robot swarm during hardware experiments.
- `simulation_data`: Training, validation and test data collected from simulations.
- `src`:
    - `anomaly_detectors`: The anomaly detector uses a neural network to predict the probability density values of the robot actions and applies the detection criterion.
    - `deployment_area`: Model of the area that the robot swarm deploys in.
    - `robots`: Model of a HERA robot.
    - `visualize`: Plots and animations.
- `trained_models`: Neural networks that are trained to predict the probability density value of a robot action given the robot's environmental context.
- `example.ipynb`: Jupyter notebook showing an example run including the anomaly detection method.
- `setup.py`: Setup file.
- `requirements.txt`: Required packages to run the notebooks.

## Installation
To install the required packages, run the following command:
```bash
pip install -r requirements.txt
```

## License
This repository is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
