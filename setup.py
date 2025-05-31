from setuptools import setup, find_packages


def parse_requirements(filename: str) -> list[str]:
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="detect-commercials",
    version="1.0.1",
    packages=find_packages(exclude=["tests*", "test*"]),
    python_requires=">=3.10, <3.13",
    install_requires=parse_requirements("requirements.txt"),
    entry_points={
        "console_scripts": [
            "detect-commercials=detect_commercials.main:main",
        ],
    },
)
