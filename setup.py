from setuptools import setup, find_packages

setup(
    name="flame_feature_extractor",
    version="0.0.1",
    description="Flame Feature Extraction",
    author="Nabarun Goswami",
    author_email="nabarungoswami@mi.t.u-tokyo.ac.jp",
    packages=find_packages(),
    package_dir={"": "."},
    install_requires=[
        "mediapipe",
        "scikit-image",
        "onnx",
        "onnxruntime",
    ],
)
