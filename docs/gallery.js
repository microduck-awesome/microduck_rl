const videos = [...document.querySelectorAll('video')];
const rate = document.querySelector('#rate');
function applyRate() {
  for (const video of videos) {
    video.defaultPlaybackRate = Number(rate.value);
    video.playbackRate = Number(rate.value);
  }
}
rate.addEventListener('change', applyRate);
for (const video of videos) video.addEventListener('loadedmetadata', applyRate);
document.querySelector('#pause').addEventListener('click', () => videos.forEach(video => video.pause()));
