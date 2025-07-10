function parseDurationToSeconds(duration) {
    // Convert duration like "1:23:45" or "12:34" to seconds
    let parts = duration.split(':').map(Number).reverse();
    let seconds = 0;
    for (let i = 0; i < parts.length; i++) {
        seconds += parts[i] * Math.pow(60, i);
    }
    return seconds;
}

let videos = [...document.querySelectorAll('ytd-rich-grid-media')].map(el => {
    // Use aria-label or a deeper title element
    let aTag = el.querySelector('a#thumbnail') || el.querySelector('a.ytd-thumbnail');
    let hoverCard = el.querySelector('#video-title');
    
    let url = aTag?.href ? aTag.href : '';
    let videoId = url.includes("watch?v=") ? url.split("watch?v=")[1]?.substring(0, 11) : '';
    
    let durationText = el.querySelector('ytd-thumbnail-overlay-time-status-renderer span')?.textContent.trim() || '';
    let durationSeconds = parseDurationToSeconds(durationText);
    
    // Try to get the title either from aria-label or hoverCard text
    let title = hoverCard?.textContent.trim() || aTag?.getAttribute('aria-label')?.split(' by ')[0] || '';

    return {
        videoId,
        url,
        duration: durationText,
        durationSeconds,
        title
    };
});

console.table(videos);
