import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pacltune",
    version="0.3.0",
    author="Georg Steinbuss",
    author_email="ck263@uni-heidelberg.de",
    description="Tuning of image patch based classifiers in medical applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    setup_requires=['wheel'],
    install_requires=[
          'psutil',
          'tabulate',
          'pyyaml',
          'numpy',
          'pandas',
          # 'tensorflow >= 2.3.1',
          'tensorflow_addons',
          'albumentations'
      ],
    python_requires='>=3.6',
)
