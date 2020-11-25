from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name="mtmct-reid",
    version="0.1.0",
    description="State-of-the-art top model for person re-identification in Multi-camera Multi-Target Tracking. Benchmarked on Market-1501 and DukeMTMTC-reID datasets.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Suraj Donthi",
    author_email="",
    url="https://github.com/SurajDonthi/Clean-ST-ReID-Multi-Target-Multi-Camera-Tracking",
    license="MIT License",
    packages=find_packages(where='mtmct_reid'),
    install_requires=[
        'pytorch>=1.5.0',
        'torchvision>=0.6.0',
        'pytorch-lightning==0.9.0'
        'tensorboard==2.2.0'
    ],
    include_package_data=True,
    platforms=["any"],
    python_requires=">3.5.2",
    entry_points={
        'console_scripts': ['mtmct-reid=mtmct_reid.main']
    }
)
