from setuptools import find_packages, setup

package_name = "argus_operator_ui"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/operator_ui.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="libargus_demo maintainers",
    maintainer_email="maintainer@example.com",
    description="PC operator UI for Argus vision and Qwen.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={"console_scripts": ["argus_operator_ui = argus_operator_ui.main:main"]},
)
