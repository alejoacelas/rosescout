"""Setup configuration for RoseScout package."""
from setuptools import setup, find_packages

setup(
    name="rosescout",
    version="0.1.0",
    description="A Streamlit-based search interface for Gemini API with modular tool integration",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "streamlit",
        "google-genai",
        "langfuse",
        "googlemaps",
        "httpx",
        "instructor",
        "openai",
        "python-dotenv"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)