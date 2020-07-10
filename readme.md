# gym-auv

Python simulation framework for Collision Avoidance for Unmanned Surface Vehicle using Deep Reinforcement Learning

## Background

This Python package, which provides an easily expandable code framework for simulating autonomous surface vessels
in maritime environments, as well as training reinforcement learning-based AI agents to guide them, was developed as a part of my Master's thesis at the Norwegian University of Science and Technology.

Apart from the underlying simulation engine, which simulates the vessel dynamics according to well-researched manuevering theory,
as well as the functioning of a LiDAR-based sensor suite for distance measurements.
it also provides easy-to-use rendering in both 2D and 3D for debugging and showcasing purposes. Implemented as an extention of the OpenAI gym toolkit, it offers an easy-to-use interface for training state-of-the-art deep reinforcement learning algorithms for guiding the vessel.

The research paper [Taming an Autonomous Surface Vehicle for Path Following and Collision Avoidance Using Deep Reinforcement Learning (2020)](https://ieeexplore.ieee.org/document/9016254?fbclid=IwAR3obkbKJcbA2Jrn3nqKp7iUD_MAag01YSCm3liaIYJN7xN9enzdHUA0Ma8) gives a comprehensive overview of what the package is intended for.

>  In this article, we explore the feasibility of applying proximal policy optimization, a state-of-the-art deep reinforcement learning algorithm for continuous control tasks, on the dual-objective problem of controlling an underactuated autonomous surface vehicle to follow an a priori known path while avoiding collisions with non-moving obstacles along the way. The AI agent, which is equipped with multiple rangefinder sensors for obstacle detection, is trained and evaluated in a challenging, stochastically generated simulation environment based on the OpenAI gym Python toolkit. Notably, the agent is provided with real-time insight into its own reward function, allowing it to dynamically adapt its guidance strategy. Depending on its strategy, which ranges from radical path-adherence to radical obstacle avoidance, the trained agent achieves an episodic success rate close to 100%.

## Getting Started

After downloading the package, install the required Python libraries via

```
pip install -r requirements.py
```

The run script can be executed with the -h flag for a comprehensive overview of the available usage modes.

## Author

* **Eivind Meyer** - [EivMeyer](https://github.com/EivMeyer)

## Screenshots

![3D Rendering](https://i.imgur.com/KD0TqZW.png)

![2D Rendering](https://i.imgur.com/dBQOWYT.png)
