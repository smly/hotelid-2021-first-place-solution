import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="peko",
    version="0.0.8",
    author="smly",
    author_email="smly@users.noreply.github.com",
    description="AH↓HA↑HA↑HA↑HA↑",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/smly",
    packages=setuptools.find_packages("."),
    entry_points={
        # "console_scripts": ["peko=peko.cmd:main"]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.6',
)
