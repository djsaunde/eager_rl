from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

version = '0.1'

setup(
    name='eager_rl',
    version=version,
    description='Tensorflow implementations of simple RL algorithms / environments.',
    license='MIT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='http://github.com/djsaunde/eager_rl',
    author='Daniel Saunders',
    author_email='danjsaund@gmail.com',
    packages=find_packages(),
    zip_safe=False,
    download_url='https://github.com/djsaunde/eager_rl/archive/%s.tar.gz' % version,
    install_requires=[
        'numpy>=1.14.2', 'tensorflow-gpu>=1.12.0', 'matplotlib>=2.1.0'
    ],
)
