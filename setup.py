import setuptools

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setuptools.setup(
    name="cdft",
    version="0.3",
    author="Lucas J. dos Santos", 
    author_email="lucasantos.318@gmail.com", 
    description="A cDFT package in Python",
    # long_description=long_description,
    long_description_content_type="text/markdown", 
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3", 
        "Operating System :: OS Independent",
    ],
)
