import os
import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install
import sys


class CustomInstallCommand(install):
    def run(self):
        # Run the standard install
        install.run(self)
        # Custom post-install script
        self.execute_post_install_script()

    def execute_post_install_script(self):
        # Get the site-packages path
        site_packages_path = next(p for p in sys.path if "site-packages" in p)
        print(site_packages_path)
        package_path = os.path.join(site_packages_path, "flame_feature_extractor")
        print(package_path)

        # Define the shell commands with the correct paths
        commands = [
            f"wget https://github.com/xg-chu/lightning_track/releases/download/resources/resources.tar -O {site_packages_path}/resources.tar",
            f"tar -xvf {site_packages_path}/resources.tar -C {site_packages_path}",
            f"mv {site_packages_path}/resources/emoca/* {package_path}/feature_extractor/emoca/assets/",
            f"mv {site_packages_path}/resources/FLAME/* {package_path}/renderer/assets/",
            f"mv {site_packages_path}/resources/mica/* {package_path}/feature_extractor/mica/assets/",
            # f"rm -r {site_packages_path}/resources/"
        ]
        os.makedirs(f"{package_path}/feature_extractor/emoca/assets")
        os.makedirs(f"{package_path}/renderer/assets")
        os.makedirs(f"{package_path}/feature_extractor/mica/assets")

        # Execute each command and print output
        for command in commands:
            subprocess.call(command, shell=True)


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
    cmdclass={
        "install": CustomInstallCommand,
    },
)
