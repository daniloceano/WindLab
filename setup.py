from setuptools import setup, find_packages

setup(
    name='windlab',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'xarray',
        'matplotlib'
    ],
    entry_points={
        'console_scripts': [
            # Adicione qualquer comando de linha de comando que queira registrar aqui
        ],
    },
    author='Danilo Couto de Souza',
    author_email='danilo.oceano@gmail.com',
    description='Pacote para manipulação e análise de dados de LIDAR (WindCube, Zephyr, e.g.).',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://codigo-externo.petrobras.com.br/tc_usp_iag_renewables/readwindcube.git',  # Altere para o repositório correto
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
    ],
    python_requires='>=3.10',
)