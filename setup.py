import setuptools

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setuptools.setup(
    name="cdft",
    version="0.0.1",
    author="Lucas J. dos Santos", 
    author_email="lucasjs@eq.ufrj.br", 
    description="A cDFT-PCSAFT package in Python",
    # long_description=long_description,
    long_description_content_type="text/markdown", 
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3", 
        "Operating System :: OS Independent",
    ],
)