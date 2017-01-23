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

Get Authorization Keys
======================

Microsoft/Oxford Emotion API `Reference <https://dev.projectoxford.ai/docs/services/5639d931ca73072154c1ce89>`_

- Place your subscription key in config.json: "Ocp-Apim-Subscription-Key: {subscription key}"

Imgur API `Reference <https://api.imgur.com/endpoints>`_

Getting Started
===============

Clone repository:

.. code-block:: python

    git clone https://github.com/JustinShenk/party-pi.git
    cd party-pi

Install dependencies:

.. code-block:: python

    pip install -r requirements.txt

Change ``config.json.example`` to ``config.json`` and add your Emotion_API and Imgur keys. Then start the game:

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
