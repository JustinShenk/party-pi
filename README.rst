Party Pi
========

|license| |nbsp| |PyPi|

Emotion detection game for parties using computer vision.

Description
===========

Interactive game that prompts players to show an emotion (eg, "Show surprise"), and displays the winner. Can be used on raspberry pi with piCamera module.

Demo
====

`Screenshot <https://www.partypi.net/img/demo.png>`_

`Pics from a demo at the Institute of Cognitive Science Christmas Party <https://coxi.partypi.net>`_

Install
=======
Install OpenCV3 with python3 bindings

Get Authorization Keys
======================

`Microsoft/Oxford Emotion API Reference <https://dev.projectoxford.ai/docs/services/5639d931ca73072154c1ce89>`_

- Place your subscription key in config.json: "Ocp-Apim-Subscription-Key: {subscription key}"


Getting Started
===============

Clone repository:

.. code-block:: python

    git clone https://github.com/JustinShenk/party-pi.git
    cd party-pi

Install dependencies:

.. code-block:: python

    pip install -r requirements.txt

If using Ubuntu, install tkinter with `sudo apt-get install python3-tk`

.. code-block:: python

    python main.py

Additional (optional)  arguments: ``--picam`` for piCamera module and ``--resolution`` to specify resolution.

Select Easy or Hard Mode (left or right arrow keys).

TODO
====
- Fix alignment of analyzing text
- Redesign mode selection boxes
- Add PyQT for font and sizing support
- Add Python 2 compatibility
- Add camera detection feature to recognize if raspberry pi

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
