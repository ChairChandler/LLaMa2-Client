from setuptools import setup

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setup(
    name='llama2-client',
    version='1.0.0',
	author='Adam Lewandowski',
	author_email='adam_lewandowski_1998@outlook.com',
	description='Web based client for LLaMa2 model.',
	long_description = long_description,
    long_description_content_type = "text/markdown",
	license='MIT License',
	keywords='llama2 llama2-client client-llama2',
	url='https://github.com/ChairChandler/LLaMa2-Client',
	package_dir={'': 'src'},
	packages=['llama2_client'],
	python_requires='>=3.11',
	install_requires=[
		'streamlit~=1.25.0',
		'transformers~=4.31.0',
		'torch~=2.0.1',
		'accelerate~=0.21.0',
		'xformers~=0.0.20',
		'optimum~=1.11.0',
		'typing_extensions~=4.7.1'
    ],
	entry_points={
		'console_scripts': [
			'llama2-client = llama2_client:run'
		]
	},
 	classifiers=[
		'Development Status :: 5 - Production/Stable',
		'Environment :: Web Environment',
		'Intended Audience :: End Users/Desktop',
		'Intended Audience :: Science/Research',
		'License :: OSI Approved :: MIT License',
		'Programming Language :: Python :: 3.11',
		'Topic :: Scientific/Engineering :: Artificial Intelligence',
		'Operating System :: OS Independent',
	]
)