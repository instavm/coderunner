"""Setup configuration for CodeRunner package"""

from setuptools import setup, find_packages
import os

def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Zero-configuration local code execution with seamless cloud migration"

def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    requirements = []
    if os.path.exists(req_path):
        with open(req_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    # Handle version constraints
                    if ">=" in line:
                        requirements.append(line)
                    elif "==" in line:
                        requirements.append(line)
                    else:
                        requirements.append(line)
    return requirements

setup(
    name="coderunner",
    version="1.0.0",
    description="Zero-config local code execution with seamless cloud migration",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="CodeRunner Team", 
    author_email="support@coderunner.dev",
    url="https://github.com/coderunner/coderunner",
    packages=find_packages(exclude=["tests*"]),
    include_package_data=True,
    install_requires=read_requirements(),
    extras_require={
        "cloud": [
            "instavm>=1.0.0"
        ],
        "integrations": [
            "openai>=1.0.0",
            "langchain>=0.1.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "pytest-asyncio>=0.18.0",
            "black>=21.0.0",
            "isort>=5.0.0",
            "mypy>=0.900"
        ],
        "all": [
            "instavm>=1.0.0",
            "openai>=1.0.0", 
            "langchain>=0.1.0",
            "pytest>=6.0.0",
            "pytest-asyncio>=0.18.0",
        ]
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10", 
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords=[
        "code execution", 
        "ai", 
        "jupyter", 
        "docker", 
        "local development", 
        "cloud computing",
        "automation",
        "sandbox"
    ],
    entry_points={
        "console_scripts": [
            "coderunner=coderunner.cli:main",
        ],
    },
    project_urls={
        "Documentation": "https://docs.coderunner.dev",
        "Source": "https://github.com/coderunner/coderunner",
        "Tracker": "https://github.com/coderunner/coderunner/issues",
    },
)