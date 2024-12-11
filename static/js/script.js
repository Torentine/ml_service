const slider = document.getElementById('slider');
const sliderValue = document.getElementById('slider-value');
const originalImage = document.getElementById('image');
const predictionImage = document.getElementById('prediction');
slider.addEventListener('input', () => {
    const value = slider.value;
    sliderValue.textContent = value;
    predictionImage.src = `{% static 'outputs/prediction_' %}${value}.png`;
    });