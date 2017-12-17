Party Pi
========

|license| |nbsp| |PyPi|

Emotion detection party game using computer vision.

Description
===========

Interactive game that ranks players by their ability to show an emotion (eg, "Show surprise"). Can be used on raspberry pi with piCamera module or with laptop.

Demo
====

`Screenshot <https://www.partypi.net/img/demo.png>`_

`Pics from a demo at the Institute of Cognitive Science 2016 Christmas Party <http://coxi.partypi.net>`_


Getting Started
===============

Clone repository:

.. code-block:: python

    git clone https://github.com/JustinShenk/party-pi.git
    cd party-pi

Install dependencies:

.. code-block:: python

    pip3 install -r requirements.txt

If using Ubuntu, install tkinter with ``sudo apt-get install python3-tk``

.. code-block:: python

    cd src/
    python3 main.py

Additional (optional)  arguments: ``--picam`` for piCamera module and ``--slow`` to slow down the countdown.

Select Easy or Hard Mode (left or right arrow keys).

TODO
====
- Fix alignment of analyzing text
- Redesign mode selection boxes
- Add PyQT for font and sizing support
- Add Python 2 compatibility
- Add camera detection feature to recognize if piCamera

Author
======

`Justin Shenk`_ (`@JustinShenk`_) created Party Pi.

.. _Justin Shenk: https://linkedin.com/in/JustinShenk/
.. _@JustinShenk: https://github.com/JustinShenk/

License
=======

`MIT <https://github.com/JustinShenk/party-pi/blob/master/LICENSE>`_

.. |license| image:: https://img.shields.io/badge/license-MIT-blue.svg
.. |PyPi| image:: https://badge.fury.io/py/partypi.svg
    :target: https://badge.fury.io/py/partypi
    :alt: PyPi Badge
.. |nbsp| unicode:: 0xA0
   :trim:
