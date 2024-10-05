// https://editor.p5js.org/
// CTRL+C fucking CTRL+V

let dots = [];
let lines = [];

function setup() {
  createCanvas(1000, 600);
  let layers = [3, 8, 6, 8, 3];

  for (let i = 0; i < layers.length; i++) {
    let layer = [];
    let x = map(i, 0, layers.length - 1, 100, width - 100);
    for (let j = 0; j < layers[i]; j++) {
      let y = map(j, 0, layers[i] - 1, 100, height - 100);
      layer.push(new Dot(x, y));
    }
    dots.push(layer);
  }

  for (let l = 0; l < dots.length - 1; l++) {
    let current = dots[l];
    let next = dots[l + 1];
    for (let i = 0; i < current.length; i++) {
      for (let j = 0; j < next.length; j++) {
        lines.push(new Line(current[i], next[j]));
      }
    }
  }
}

function draw() {
  background(0);

  for (let i = 0; i < lines.length; i++) {
    lines[i].update();
    lines[i].show();
  }

  for (let i = 0; i < dots.length; i++) {
    for (let j = 0; j < dots[i].length; j++) {
      dots[i][j].update();
      dots[i][j].show();
    }
  }
}

class Dot {
  constructor(x, y) {
    this.x = x;
    this.y = y;
    this.val = random(1);
    this.size = 20;
  }

  update() {
    this.val = noise(frameCount * 0.01 + this.x * 0.01) * 2 - 1;
  }

  show() {
    noStroke();
    fill(0, 200, 255, 80);
    ellipse(this.x, this.y, this.size + 15);

    fill(0, 200, 255);
    ellipse(this.x, this.y, this.size);

    fill(255);
    noStroke();
    textSize(12);
    textAlign(CENTER, CENTER);
    text(nf(this.val, 1, 2), this.x, this.y);
  }
}

class Line {
  constructor(fromDot, toDot) {
    this.from = fromDot;
    this.to = toDot;
    this.weight = random(-1, 1);
  }

  update() {
    this.weight = noise(frameCount * 0.01 + this.from.x * 0.01 + this.to.y * 0.01) * 2 - 1;
  }

  show() {
    strokeWeight(map(abs(this.weight), 0, 1, 1, 3));
    stroke(this.weight > 0 ? 'rgba(0,255,255,0.7)' : 'rgba(255,50,50,0.7)');
    line(this.from.x, this.from.y, this.to.x, this.to.y);
  }
}
