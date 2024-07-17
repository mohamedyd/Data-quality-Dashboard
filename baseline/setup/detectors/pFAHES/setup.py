import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="myfahes",
    version="1.0",
    author="Hakim Qahtan",
    author_email="abdulhakim.qahtan@gmai.com",
    description="FAHES: A Disguised Missing Values Detector",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qahtanaa/pFAHES",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
