from setuptools import setup, find_packages

setup(
    name="three_joint_arm_rl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "numpy",
        "torch",
        "matplotlib",
        "pandas"
    ],
    entry_points={
        "console_scripts": [
            "train-arm = src.train:train",
            "evaluate-arm = src.evaluate:evaluate"
        ]
    },
)
