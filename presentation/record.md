# Recording the 5-minute video

Format: prerecorded video of the slide deck with both of us narrating, then 2
minutes live Q&A in class. Presentation slot: Tue Dec 1, 9:51-9:58 AM.

## Setup (macOS)

1. Open `presentation.pdf` in Preview (or Skim) and go fullscreen:
   View > Enter Full Screen. Arrow keys / click advance slides.
2. Open QuickTime Player > File > New Screen Recording. Choose "Record
   Selected Portion" and frame the slide area, or record the whole screen.
   In the Options menu pick the built-in microphone (or an external mic if
   available) and turn mouse clicks off.
3. Do a 20-second test clip first to check audio levels and that the cursor
   is not distracting.

## Recording

- Rehearse once with a timer. Aim for 4:45-5:00, never over 5:00.
- Read notes.md sections per slide; the hand-off lines mark who speaks.
- One take per speaker half is fine: record both halves in one session, or
  record separately and join clips in QuickTime (Edit > Trim, then drag clips
  together in iMovie if needed).
- If a take goes badly, re-record immediately while the phrasing is fresh.

## Export and upload

- QuickTime saves .mov. To get mp4: File > Export As > 1080p produces .mov;
  convert with `afconvert`/`ffmpeg` (`ffmpeg -i in.mov -c:v h264 -c:a aac
  presentation.mp4`) or just upload the .mov, YouTube accepts it.
- Upload as an **unlisted YouTube video** (or a Drive file with link sharing
  on). Title it something plain like "CS 566 Project - Monocular Depth".
- Put the link on the project webpage (webpage/, link target next to the
  reports section). The slide PDF is already at
  `webpage/assets/presentation.pdf`.

## Checklist before submitting

- [ ] Video is 5:00 or under.
- [ ] Audio is audible on laptop speakers at 50% volume.
- [ ] Slides are legible fullscreen (they are 16:9 already).
- [ ] Link opens in a private/incognito window.
- [ ] Link added to the webpage.
