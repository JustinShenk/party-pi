/*
 *  Copyright (c) 2015 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree.
 *  Modified from https://github.com/webrtc/samples/blob/gh-pages/src/content/getusermedia/canvas/js/main.js
 */

'use strict';

// Put variables in global scope to make them available to the browser console.
var video = document.querySelector('video');
var canvas = window.canvas = document.querySelector('canvas');
canvas.width = 640;
canvas.height = 480;

var button = document.querySelector('button');
button.onclick = function() {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').
    drawImage(video, 0, 0, canvas.width, canvas.height);
};

var constraints = {
  audio: false,
  video: {
    width: {
      exact: 640
    },
    height: {
      exact: 480
    },
    facingMode: "user"
  }
};

function handleSuccess(stream) {
  window.stream = stream; // make stream available to browser console
  video.srcObject = stream;

  video.onloadedmetadata = function(e) {
    var beginStreaming = true;
    if (beginStreaming) {
      video.style.display = "block";
    }
    video.play();
    $('#play').show();
    if (!playButtonInitialized) {
      initPlayButton();
    }
  };
}

function handleError(error) {
  console.log('navigator.getUserMedia error: ', error);
  // showManualUpload();
}

navigator.mediaDevices.getUserMedia(constraints).
    then(handleSuccess).catch(handleError);
