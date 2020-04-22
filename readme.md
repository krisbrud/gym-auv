# gym-auv

Python simulation framework for Collision Avoidance for Unmanned Surface Vehicle using Deep Reinforcement Learning

## Background

This Python package, which provides an easily expandable code framework for simulating autonomous surface vessels
in maritime environments, was developed as a part of my Master's thesis at NTNU, Norway.

Apart from the underlying simulation engine, which simulates the vessel dynamics according to well-researched manuevering theory,
as well as the functioning of a LiDAR-based sensor suite for distance measurements.
it also provides easy-to-use rendering in both 2D and 3D for debugging and showcasing purposes.

The research paper [Taming an Autonomous Surface Vehicle for Path Following and Collision Avoidance Using Deep Reinforcement Learning (2020)](https://ieeexplore.ieee.org/document/9016254?fbclid=IwAR3obkbKJcbA2Jrn3nqKp7iUD_MAag01YSCm3liaIYJN7xN9enzdHUA0Ma8) gives a comprehensive overview of what the package is intended for.

![3D Rendering](https://i.imgur.com/KD0TqZW.png)

![2D Rendering](https://i.imgur.com/dBQOWYT.png)

## Getting Started

After downloading the package, install the required Python libraries via

```
pip install -r requirements.py
```

The run script can be executed with the -h flag for a comprehensive overview of the available usage modes.

## Author

* **Eivind Meyer** - *Initial work* - [EivMeyer](https://github.com/EivMeyer)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
