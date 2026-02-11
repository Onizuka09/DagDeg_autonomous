from setuptools import find_packages, setup
import os
import glob

package_name = "dagdeg_nav"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # Include launch files
        ("share/" + package_name + "/launch", ["launch/display_robot.launch.py"]),
        ("share/" + package_name + "/launch", ["launch/real_robot.launch.py"]),
        # Include URDF files
        ("share/" + package_name + "/urdf", ["urdf/dagdeg_bot.urdf"]),
        # Include RViz config files
         ("share" + package_name + "/config", ["config/robot_display.rviz"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="moktar",
    maintainer_email="smokthar925@gmail.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "zetta_bridge = dagdeg_nav.zetta_bridge:main",
            "nav_cmd_pub = dagdeg_nav.nav_command_pub:main",

        ],
    },
)
