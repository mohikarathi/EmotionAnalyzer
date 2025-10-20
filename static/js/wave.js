const canvas = document.getElementById('waveCanvas');
const ctx = canvas.getContext('2d');

function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

let t = 0;

function drawWave() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const centerY = canvas.height / 2;

    // Layer 1: Main wave
    ctx.beginPath();
    for (let x = 0; x < canvas.width; x++) {
        const y = centerY + Math.sin(x * 0.01 + t * 0.5) * 60 + Math.cos(x * 0.005 + t) * 30;
        x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)'; // darker alpha
    ctx.lineWidth = 4;
    ctx.stroke();

    // Layer 2: Secondary wave
    ctx.beginPath();
    for (let x = 0; x < canvas.width; x++) {
        const y = centerY + Math.sin(x * 0.02 + t * 0.75) * 20;
        x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)'; // darker alpha
    ctx.lineWidth = 2;
    ctx.stroke();

    t += 0.015;
    requestAnimationFrame(drawWave);
}

drawWave();
