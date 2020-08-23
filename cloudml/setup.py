from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ["matplotlib","pandas","yfinance","lxml","google-cloud-pubsub==1.5.0","alpaca-trade-api","get-all-tickers","ta"]

setup(
  name='stock',
  version='1.0',
  install_requires=REQUIRED_PACKAGES,
  packages=find_packages(),
  include_package_data=True,
  description='AI Platform Boston samples'
)
