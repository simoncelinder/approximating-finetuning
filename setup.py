from setuptools import setup

optional_dependencies = {
    'test': [
        'pytest==7.2.0',
        'pytest-cov==4.0.0',
        'pytest-mock==3.10.0'
    ],
    'plot': [
        'plotly==5.11.0',
        'cufflinks==0.17.3',
        'chart-studio==1.1.0',
        'seaborn==0.12.2',
    ]
}

setup(
    name='approximating-finetuning',
    version='0.0.1',
    python_version='3.10',
    description='Blending of language models',
    install_requires=[
        'openai==0.25.0',
        'jupyterlab==3.5.2',
        'jupyterlab==3.5.2',
        'jupytext==1.14.4',
        'pandas==1.5.2',
        'numpy==1.24.1',
        'pyarrow==11.0.0',
        'optuna==3.1.0',
        'scipy==1.10.0',
        'joblib==1.2.0',
        'transformers==4.27.4',
    ],
    extras_require=optional_dependencies,
    tests_require=['pytest'],
    zip_safe=False,
)
