Party Pi
########

Emotion detection game for parties using computer vision.

Description
===========

Interactive game that prompts players to show an emotion (eg, "Show surprise"), and displays the winner. Can be used on raspberry pi with piCamera module.

Demo
====
.. image:: partypi/demo.png
   
Install
=======
Install OpenCV3 with python3 bindings

Mac
---

Brew

.. code-block:: python
    brew install opencv3 --with-python3 --with-contrib

Anaconda

.. code-block:: python
    conda install -c menpo opencv3=3.1.0

Ubuntu
------

`Installing`_ OpenCV3 on Ubuntu

.. _Installing: http://www.pyimagesearch.com/2015/07/20/install-opencv-3-0-and-python-3-4-on-ubuntu/

Windows
-------

`Installing`_ OpenCV3 on Windows

.. _Installing: https://www.solarianprogrammer.com/2016/09/17/install-opencv-3-with-python-3-on-windows/

Get Authorization Keys
======================

Microsoft/Oxford Emotion API `Reference`_

.. _Reference: https://dev.projectoxford.ai/docs/services/5639d931ca73072154c1ce89

- Place your subscription key in config.json: "Ocp-Apim-Subscription-Key: {subscription key}"

Imgur API Reference_
.. _Reference: https://api.imgur.com/endpoints

Getting Started
===============

Clone repository

.. code-block:: python

    git clone https://github.com/JustinShenk/party-pi.git
    cd party-pi

Install dependencies

.. code-block:: python
    pip install -r requirements.txt

Change config.json.example to config.json and add your Emotion_API and Imgur keys.

.. code-block:: python
    python main.py

Additional arguments: `--picam` for piCamera module and `--resolution` to specify resolution.

Select Easy or Hard Mode (left or right arrow keys)

TODO
====
 - [ ] Fix alignment of analyzing text
 - [ ] Redesign mode selection boxes
 - [ ] Add PyQT for font and sizing support
 - [ ] Add Python 2 compatibility

Author
======

`Justin Shenk`_ (`@JustinShenk`_) created Party Pi.
.. _Justin Shenk: https://linkedin.com/in/JustinShenk/
.. _@JustinShenk: https://github.com/JustinShenk/
