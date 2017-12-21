Party Pi
========

|license| |nbsp| |PyPi|

Emotion detection game using computer vision.

Description
===========

Interactive game that ranks players by their ability to show an emotion (eg, "Show surprise"). Can be used on Raspberry Pi with piCamera module or with laptop.

Emotion detection is accomplished using an Inception-based neural network trained in TensorFlow. Face detection is accomplished using OpenCV's Haar cascade.

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

Awards, Press, Demos
====================

Party Pi was `featured <https://software.intel.com/en-us/blogs/2017/08/23/intel-developer-mesh-editor-s-picks-august-2017>`_ in the Intel Developer Mesh Editor's Picks (August 2017), was awarded the 2017 Intel Hack Challenge prize, and the technology was demonstrated at Intel's AI booth at NIPS 2017. It was demoed for students and faculty present at the 2016 and 2017 University of Osnabrueck Institute of Cognitive Science Christmas Parties.

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
