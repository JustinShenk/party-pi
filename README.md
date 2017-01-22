# Party Pi
Emotion detection game for parties.

## Description
Interactive game that prompts players to show an emotion (eg, "Show surprise"), and displays the winner. 

## Demo
![happiness](https://www.github.com/JustinShenk/party-pi/blob/master/demo.png)

## Install
Install OpenCV3 with python3 bindings

### Mac

#### Brew
`brew install opencv3 --with-python3 --with-contrib`

#### Anaconda
conda install -c menpo opencv3=3.1.0

### Ubuntu

[Installing OpenCV3 on Ubuntu](http://www.pyimagesearch.com/2015/07/20/install-opencv-3-0-and-python-3-4-on-ubuntu/)

### Windows

[Installing OpenCV3 on Windows](https://www.solarianprogrammer.com/2016/09/17/install-opencv-3-with-python-3-on-windows/)

## Get authorization keys

[Microsoft/Oxford Emotion API](https://dev.projectoxford.ai/docs/services/5639d931ca73072154c1ce89)
- Place your subscription key in auth.json: "Ocp-Apim-Subscription-Key: {subscription key}"

[Imgur API](https://api.imgur.com/endpoints)

## Getting Started

Clone repository - `git clone https://github.com/JustinShenk/party-pi.git`
`cd party-pi`

Change auth.json.example to auth.json and add your Emotion_API and Imgur keys.

`python main.py`

Select Easy or Hard Mode (left or right arrow keys)

## TODO
 - [ ] Fix alignment of analyzing text
 - [ ] Redesign mode selection boxes
 - [ ] Add PyQT

## License

Mit

## Website

[partypi.net](https://partypi.net)

## Author

[Justin Shenk](https://github.com/justinshenk/)