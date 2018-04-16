var isMobile;
var n_rounds_played = 0;
var showGallery = false;

if (/Android|webOS|iPhone|iPad|iPod|BlackBerry|BB|PlayBook|IEMobile|Windows Phone|Kindle|Silk|Opera Mini/i.test(navigator.userAgent)) {
  //alert("Support for mobile devices is still in developmental stage. For the best experience try desktop Chrome.");
  isMobile = true;
} else {
  isMobile = false;
  $("#text").css('display', 'block');
  $("#play").css('display', 'block');
  $("#videoElement").css('display', 'block');
}

var emotionArray = [
  'happy',
  'angry',
  'sad',
  'disgust',
  'fear',
  'surprise',
  'neutral'
];

function saveMe(img_id) {
  if (!typeof img_id === 'string') {
    img_id = img_id.id;
  }
  var img_src = $("#" + img_id).attr("src");
  var a = document.createElement('a');
  a.href = img_src;
  var filename = "party_pi_" + String(img_id) + ".jpg";
  console.log(filename);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

function moveToGallery() {
  // Move photo to gallery
  $('#gallery').css("display", "block");
  $('#output img:visible').each(function(i) {
    // Create card for image
    var img_src = $(this).attr('src');

    $(this).remove();

    var galleryLength = $(".card-img-top").length;
    var img_filename = "img_" + galleryLength;
    var myCard = $('<div class="col-md-4"><div class="card mb-4 box-shadow"><img id="' + img_filename + '" class="card-img-top" src="' + img_src +
      '" alt="Card image cap"><div class="card-body"><div class="d-flex justify-content-between align-items-center"><div class="btn-group"><button type="button" id="btn_' + galleryLength +
      '" class="btn btn-sm btn-outline-secondary view-btn" onclick="saveMe(&apos;img_' + galleryLength + '&apos;)"><i class="fas fa-download"></i></button></div><small class="text-muted"></small></div></div></div>');
    myCard.prependTo('#galleryImages');
  });
}

// Function to check orientation of image from EXIF metadatas and draw canvas
function orientation(img, canvas) {

  // Set variables
  var ctx = canvas.getContext("2d");
  ctx.restore();
  var exifOrientation = '';
  var width = img.width,
    height = img.height;

  console.log("480 * " + img.width + "/" + img.height + " x " + height);
  // Check orientation in EXIF metadatas
  EXIF.getData(img, function() {
    var allMetaData = EXIF.getAllTags(this);
    exifOrientation = allMetaData.Orientation;
    console.log('Exif orientation: ' + exifOrientation);
  });

  // set proper canvas dimensions before transform & export
  if (jQuery.inArray(exifOrientation, [5, 6, 7, 8]) > -1) {
    console.log("switching");
    canvas.width = height;
    canvas.height = width;
  } else {
    canvas.width = width;
    canvas.height = height;
  }

  // transform context before drawing image
  switch (exifOrientation) {
    case 2:
      ctx.transform(-1, 0, 0, 1, width, 0);
      break;
    case 3:
      ctx.transform(-1, 0, 0, -1, width, height);
      break;
    case 4:
      ctx.transform(1, 0, 0, -1, 0, height);
      break;
    case 5:
      ctx.transform(0, 1, 1, 0, 0, 0);
      break;
    case 6:
      ctx.transform(0, 1, -1, 0, height, 0);
      break;
    case 7:
      ctx.transform(0, -1, -1, 0, height, width);
      break;
    case 8:
      ctx.transform(0, -1, 1, 0, 0, width);
      break;
    default:
      ctx.transform(1, 0, 0, 1, 0, 0);
  }

  // Draw img into canvas
  ctx.drawImage(img, 0, 0, width, height);
}

var fileUpload = $("#imageInput");

function sendFile(dataURL) {
  // var ctx = canvas.getContext("2d");
  // var dataURL = canvas.toDataURL('image/jpeg', 0.7);
  // console.log("canvas: " + canvas.width + " " + canvas.height);
  var fd = new FormData(document.forms[0]);
  fd.append("imageBase64", dataURL);
  fd.append("emotion", currentEmotion);

  $("#spinner").addClass('loader');
  $("#placeholder").show();
  // send image to API
  $.ajax({
    type: "POST",
    url: "{{ url_for('image') }}",
    data: fd,
    async: true,
    cache: false,
    headers: {
      "cache-control": "no-cache"
    },
    processData: false,
    contentType: false,
    timeout: 10000
  }).done(function(data) {
    // Add photo to top
    var img = document.createElement('img');
    img.setAttribute('id', 'photo');
    img.src = "{{ url_for('static', filename='{{data.photoPath}}' }}";
    $("#output").prepend(img);
    $("#placeholderImage").hide();
    $("#spinner").removeClass('loader');
    var emotion = getRandomEmotion();
    currentEmotion = emotion;
    $("#mobilePrompt").text("Show " + currentEmotion + "_");
  });
}

function resetOrientation(srcBase64, srcOrientation, callback) {
  var img = new Image();

  img.onload = function() {
    var width = img.width,
      height = img.height,
      canvas = document.createElement('canvas'),
      ctx = canvas.getContext("2d");

    // set proper canvas dimensions before transform & export
    if (4 < srcOrientation && srcOrientation < 9) {
      canvas.width = height;
      canvas.height = width;
    } else {
      canvas.width = width;
      canvas.height = height;
    }

    // transform context before drawing image
    switch (srcOrientation) {
      case 2:
        ctx.transform(-1, 0, 0, 1, width, 0);
        break;
      case 3:
        ctx.transform(-1, 0, 0, -1, width, height);
        break;
      case 4:
        ctx.transform(1, 0, 0, -1, 0, height);
        break;
      case 5:
        ctx.transform(0, 1, 1, 0, 0, 0);
        break;
      case 6:
        ctx.transform(0, 1, -1, 0, height, 0);
        break;
      case 7:
        ctx.transform(0, -1, -1, 0, height, width);
        break;
      case 8:
        ctx.transform(0, -1, 1, 0, 0, width);
        break;
      default:
        break;
    }

    // draw image
    alert("before draw " + img.width + ' ' + img.height);
    ctx.drawImage(img, 0, 0);

    // export base64
    callback(canvas.toDataURL());
  };
  img.src = srcBase64;
  return img;
}

function readURL(input) {

  if (input.files && input.files[0]) {
    var reader = new FileReader();

    reader.onloadend = function(e) {
      srcOrientation = '';
      var img = document.createElement('img');
      img.src = e.target.result;
      console.log("exif?");
      EXIF.getData(img, function() {
        var allMetaData = EXIF.getAllTags(this);
        exifOrientation = allMetaData.Orientation;
        console.log('Exif orientation: ' + srcOrientation);
      });
      img = resetOrientation(e.target.result, srcOrientation, function() {});
      var dataURL = img.src;
      sendFile(dataURL);
    };
    reader.readAsDataURL(input.files[0]);
  }
}

fileUpload.on('change', function() {
  console.log("reading");
  readURL(this);
  if (n_rounds_played > 0) { // leave the first image at the top
    moveToGallery(); // move }old images to gallery
  }
  n_rounds_played++;
});


// end image upload

function getRandomEmotion() {
  var randomNumber = Math.floor(Math.random() * emotionArray.length);
  return emotionArray[randomNumber];
}

function errorMessage(message, e) {
  console.error(message, typeof e == 'undefined' ? '' : e);
}

var currentEmotion = 'happy';

function getWebcam() {
  if (navigator.mediaDevices === undefined) {
    navigator.mediaDevices = {};
  }

  // Here, we will just add the getUserMedia property if it's missing.
  if (navigator.mediaDevices.getUserMedia === undefined) {
    navigator.mediaDevices.getUserMedia = function(constraints) {

      // First get ahold of the legacy getUserMedia, if present
      var getUserMedia = navigator.webkitGetUserMedia || navigator.mozGetUserMedia;

      // Some browsers just don't implement it - return a rejected promise with an error
      // to keep a consistent interface
      if (!getUserMedia) {
        return Promise.reject(new Error('getUserMedia is not implemented in this browser'));
      }

      // Otherwise, wrap the call to the old navigator.getUserMedia with a Promise
      return new Promise(function(resolve, reject) {
        getUserMedia.call(navigator, constraints, resolve, reject);
      });
    }
  }
}

function startWebcam() {
  if (location.protocol === 'https:') {
    // navigator.getUserMedia = navigator.getUserMedia || navigator.mediaDevices.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia || navigator.oGetUserMedia;

    // Prefer camera resolution nearest to 640x480.
    var constraints = {
      audio: false,
      video: {
        width: {
          exact: 640
        },
        height: {
          exact: 480
        }
      }
    };

    navigator.mediaDevices.getUserMedia(constraints)
      .then(function(stream) {
        $("#videoElement").show();
        console.log("Hide webcam");
        $('#imageInput').filestyle('destroy');
        $('#imageInput').hide();
        console.log("Destroy #imageInput");

        var video = $("#videoElement").get(0);
        // Older browsers may not have srcObject
        if ("srcObject" in video) {
          video.srcObject = stream;
        } else {
          // Avoid using this in new browsers, as it is going away.
          video.src = window.URL.createObjectURL(stream);
        }
        video.onloadedmetadata = function(e) {
          video.play();
        };
      })
      .catch(function(e) {
        var message;
        switch (e.name) {
          case 'NotFoundError':
          case 'DevicesNotFoundError':
            message = 'Please setup your webcam first.';
            break;
          case 'SourceUnavailableError':
            message = 'Your webcam is busy';
            break;
          case 'PermissionDeniedError':
          case 'SecurityError':
            message = 'Permission denied!';
            break;
          default:
            errorMessage('Reeeejected!', e);
            return;
        }
        console.log("HTML5 input not available, switching to file input.")
        errorMessage(message);
      });
  } else {
    console.log('HTTPS not found');
  }
};
// end startWebcam

function dataURItoBlob(dataURI) {
  // convert base64/URLEncoded data component to raw binary data held in a string
  var byteString;
  if (dataURI.split(',')[0].indexOf('base64') >= 0)
    byteString = atob(dataURI.split(',')[1]);
  else
    byteString = unescape(dataURI.split(',')[1]);

  // separate out the mime component
  var mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0];

  // write the bytes of the string to a typed array
  var ia = new Uint8Array(byteString.length);
  for (var i = 0; i < byteString.length; i++) {
    ia[i] = byteString.charCodeAt(i);
  }

  return new Blob([ia], {
    type: mimeString
  });
}

$(document).ready(function() {
  "use strict";

  var $output;
  var scale = 1;

  var initialize = function() {
    getWebcam();
    if (!isMobile) {
      $('#play').show();
      startWebcam();
      $("#play").click(function() {
        $("#placeholderImage").hide();
        $("#videoElement").removeAttr('hidden');
        $("#myProgress").show();
        if (showGallery) {
          // move photo from main display to gallery
          moveToGallery();
        } else {
          showGallery = true;
        }

        // Show and start webcam
        $("#text").text("Show " + currentEmotion + "_");
        // Hide play button
        $(this).hide();
        // Start timer
        move();
        // Capture image and load image
        setTimeout(captureImage, 3000);
      });
    } else { // is mobile
      $("#imageInput").show();
      console.log("Show #imageInput");
      $('#videoElement').hide();
      $('#play').hide();
      $('#myProgress').hide();
      $('#text').hide();
      $("#mobilePrompt").text("Show " + currentEmotion + "_");
      $('#mobilePrompt').show();
    }
  };

  var captureImage = function() {
    var video = $("#videoElement").get(0);
    var canvas = document.createElement("canvas");
    canvas.width = video.videoWidth * scale;
    canvas.height = video.videoHeight * scale;
    var canvasContext = canvas.getContext('2d');
    canvasContext.translate(video.videoWidth, 0);
    canvasContext.scale(-1, 1);
    canvasContext.drawImage(video, 0, 0, canvas.width, canvas.height);

    var dataURL = canvas.toDataURL('image/jpeg', 1.0);
    var fd = new FormData(document.forms[0]);
    fd.append("imageBase64", dataURL);
    fd.append("emotion", currentEmotion);
    $("#placeholderImage").show();
    $output = $("#output");
    $("#spinner").addClass("loader");
    video.setAttribute('hidden', 'true');
    console.log('submitting');

    $.ajax({
      type: "POST",
      url: "{{ url_for('image') }}",
      data: fd,
      processData: false,
      contentType: false,
      timeout: 10000
    }).done(function(data) {
      $("#placeholderImage").hide();
      $("#spinner").removeClass("loader");
      // Add photo to top
      var img = document.createElement('img');
      img.src = data.photoPath;
      $output.prepend(img);
      $("#myProgress").hide();
      $("#play").show();
    });
  }; // end captureImage

  $(initialize);
});

<!-- PROGRESS BAR -->
function move() {
  var elem = document.getElementById("countdownBar");
  var width = 1;
  var id = setInterval(frame, 30);

  function frame() {
    if (width >= 100) {
      clearInterval(id);
    } else {
      width++;
      elem.style.width = width + '%';
    }
  }
}
