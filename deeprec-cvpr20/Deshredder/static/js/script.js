// toggle showing or not showing the about section
function toggleAbout() {
    const aboutInfo = document.getElementById('about-info');
    const toggleIcon = document.getElementById('toggle-icon');
    if (aboutInfo.style.display === 'none' || aboutInfo.style.display === '') {
        aboutInfo.style.display = 'block';
        toggleIcon.classList.add('open');
    } else {
        aboutInfo.style.display = 'none';
        toggleIcon.classList.remove('open');
    }
}