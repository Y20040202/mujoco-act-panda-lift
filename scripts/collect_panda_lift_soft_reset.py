import robosuite

from lerobot_mujoco.datasets.collect_demonstrations import main

original_make = robosuite.make


def make_with_soft_reset(env_name, **kwargs):
    if env_name == "Lift":
        kwargs["hard_reset"] = False
    return original_make(env_name, **kwargs)


if __name__ == "__main__":
    robosuite.make = make_with_soft_reset
    main()
