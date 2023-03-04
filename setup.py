from setuptools import setup

package_name = 'squeeze_wrench_estimator'
submodules = "squeeze_wrench_estimator"

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name,submodules],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='napster',
    maintainer_email='aravindadhith@asu.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        	'wrench_estimator = squeeze_wrench_estimator.estimator:main'
        ],
    },
)
