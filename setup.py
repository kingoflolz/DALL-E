from setuptools import setup

def parse_requirements(filename):
	lines = (line.strip() for line in open(filename))
	return [line for line in lines if line and not line.startswith("#")]

setup(name='DALL-E-JAX',
        version='0.1',
        description='JAX package for the discrete VAE used for DALL·E.',
        url='http://github.com/kingololz/DALL-E-JAX',
        author='Ben Wang',
        author_email='wangben3@gmail.com',
        license='BSD',
        packages=['dall_e_jax'],
        install_requires=parse_requirements('requirements.txt'),
        zip_safe=True)
