from setuptools import find_packages, setup

package_name = 'vision_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/models',['models/traffic_light_model_best.pt']),
        ('share/' + package_name + '/models',['models/traffic_light_model.tflite'])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='moktar',
    maintainer_email='smokthar925@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'yolo_traffic_light_dector = vision_perception.yolo_traffic_light_node:main',
            'tflite_traffic_light_dector = vision_perception.tflite_traffic_light_node:main'
        ],
    },
)
