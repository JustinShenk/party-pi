from setuptools import setup, find_packages

import re
import sys


VERSION = '0.1.0' # x.y.z[-dev#]
REPOSITORY = 'https://github.com/JustinShenk/party-pi'
PACKAGES = find_packages(where='partypi')
README = ''
with open('README.rst', 'r') as f:
    README = f.read()
README = re.sub(r' _(.+): ([^(http)].+)', r' _\1: {}/blob/master/\2'.format(REPOSITORY), README)

setup(
  name = 'partypi',
  version = VERSION,
  description = 'Party Pi is a computer vision emotion detection game with OpenCV and Microsoft Oxford Emotion API.',
  long_description = README,
  author = 'Justin Shenk',
  author_email = 'shenk.justin@gmail.com',

  url = REPOSITORY,
  download_url = '{}/tarball/{}'.format(REPOSITORY, VERSION),
  packages = PACKAGES,
  classifiers = [
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Education',
      'Intended Audience :: Science/Research',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Topic :: Games/Entertainment',
      'Programming Language :: Python :: 3.5',
      'Programming Language :: Python :: 3.6',
      'Topic :: Multimedia :: Video :: Display',
  ],
  license = 'MIT',
  keywords = [
      'OpenCV', 'cv2', 'emotion', 'detection', 'game'
  ],
)
