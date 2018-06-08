function WordShuffler(holder, opt) {
  var that = this;
  var time = 0;
  this.now;
  this.then = Date.now();

  this.delta;
  this.currentTimeOffset = 0;

  this.word = null;
  this.currentWord = null;
  this.currentCharacter = 0;
  this.currentWordLength = 0;

  var options = {
    fps: 20,
    timeOffset: 5,
    textColor: '#000',
    fontSize: "50px",
    useCanvas: false,
    mixCapital: false,
    mixSpecialCharacters: false,
    needUpdate: true,
    colors: [
      '#f44336', '#e91e63', '#9c27b0',
      '#673ab7', '#3f51b5', '#2196f3',
      '#03a9f4', '#00bcd4', '#009688',
      '#4caf50', '#8bc34a', '#cddc39',
      '#ffeb3b', '#ffc107', '#ff9800',
      '#ff5722', '#795548', '#9e9e9e',
      '#607d8b'
    ]
  }

  if (typeof opt != "undefined") {
    for (key in opt) {
      options[key] = opt[key];
    }
  }

  this.needUpdate = true;
  this.fps = options.fps;
  this.interval = 1000 / this.fps;
  this.timeOffset = options.timeOffset;
  this.textColor = options.textColor;
  this.fontSize = options.fontSize;
  this.mixCapital = options.mixCapital;
  this.mixSpecialCharacters = options.mixSpecialCharacters;
  this.colors = options.colors;

  this.useCanvas = options.useCanvas;

  this.chars = [
    'A', 'B', 'C', 'D',
    'E', 'F', 'G', 'H',
    'I', 'J', 'K', 'L',
    'M', 'N', 'O', 'P',
    'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X',
    'Y', 'Z'
  ];
  this.specialCharacters = [
    '!', '§', '$', '%',
    '&', '/', '(', ')',
    '=', '?', '_', '<',
    '>', '^', '°', '*',
    '#', '-', ':', ';', '~'
  ]

  if (this.mixSpecialCharacters) {
    this.chars = this.chars.concat(this.specialCharacters);
  }

  this.getRandomColor = function() {
    var randNum = Math.floor(Math.random() * this.colors.length);
    return this.colors[randNum];
  }

  //if Canvas

  this.position = {
    x: 0,
    y: 50
  }

  //if DOM
  if (typeof holder != "undefined") {
    this.holder = holder;
  }

  if (!this.useCanvas && typeof this.holder == "undefined") {
    console.warn('Holder must be defined in DOM Mode. Use Canvas or define Holder');
  }


  this.getRandCharacter = function(characterToReplace) {
    if (characterToReplace == " ") {
      return ' ';
    }
    var randNum = Math.floor(Math.random() * this.chars.length);
    var lowChoice = -.5 + Math.random();
    var picketCharacter = this.chars[randNum];
    var choosen = picketCharacter.toLowerCase();
    if (this.mixCapital) {
      choosen = lowChoice < 0 ? picketCharacter.toLowerCase() : picketCharacter;
    }
    return choosen;

  }

  this.writeWord = function(word) {
    this.word = word;
    this.currentWord = word.split('');
    this.currentWordLength = this.currentWord.length;

  }

  this.generateSingleCharacter = function(color, character) {
    var span = document.createElement('span');
    span.style.color = color;
    span.innerHTML = character;
    return span;
  }

  this.updateCharacter = function(time) {

    this.now = Date.now();
    this.delta = this.now - this.then;

    if (this.delta > this.interval) {
      this.currentTimeOffset++;

      var word = [];

      if (this.currentTimeOffset === this.timeOffset && this.currentCharacter !== this.currentWordLength) {
        this.currentCharacter++;
        this.currentTimeOffset = 0;
      }
      for (var k = 0; k < this.currentCharacter; k++) {
        word.push(this.currentWord[k]);
      }

      for (var i = 0; i < this.currentWordLength - this.currentCharacter; i++) {
        word.push(this.getRandCharacter(this.currentWord[this.currentCharacter + i]));
      }


      if (that.useCanvas) {
        c.clearRect(0, 0, stage.x * stage.dpr, stage.y * stage.dpr);
        c.font = that.fontSize + " sans-serif";
        var spacing = 0;
        word.forEach(function(w, index) {
          if (index > that.currentCharacter) {
            c.fillStyle = that.getRandomColor();
          } else {
            c.fillStyle = that.textColor;
          }
          c.fillText(w, that.position.x + spacing, that.position.y);
          spacing += c.measureText(w).width;
        });
      } else {

        if (that.currentCharacter === that.currentWordLength) {
          that.needUpdate = false;
        }
        this.holder.innerHTML = '';
        word.forEach(function(w, index) {
          var color = null
          if (index > that.currentCharacter) {
            color = that.getRandomColor();
          } else {
            color = that.textColor;
          }
          that.holder.appendChild(that.generateSingleCharacter(color, w));
        });
      }
      this.then = this.now - (this.delta % this.interval);
    }
  }

  this.restart = function() {
    this.currentCharacter = 0;
    this.needUpdate = true;
  }

  function update(time) {
    time++;
    if (that.needUpdate) {
      that.updateCharacter(time);
    }
    requestAnimationFrame(update);
  }

  this.writeWord(this.holder.innerHTML);

  update(time);
}

var text = document.getElementById('prompt');
var shuffler = document.getElementById('play');

var pText = new WordShuffler(text, {
  textColor: '#fff',
  timeOffset: 2
});

var buttonText = new WordShuffler(shuffler, {
  textColor: 'tomato',
  timeOffset: 10
});

shuffler.addEventListener('click', function() {;
  // Get random emotion
  var emotion = getRandomEmotion();
  currentEmotion = emotion;
  pText.currentWord = 'Show ' + currentEmotion + '_            '
  pText.restart();
  buttonText.restart();
});
