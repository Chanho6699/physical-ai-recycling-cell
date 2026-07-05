from setuptools import find_packages, setup

package_name = 'recycling_cell_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rlack',
    maintainer_email='rlack@todo.todo',
    description='Vision perception skeleton (image pipeline + mock detections, ready for YOLO/ONNX)',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'vision_perception_node = recycling_cell_vision.vision_perception_node:main',
        ],
    },
)
