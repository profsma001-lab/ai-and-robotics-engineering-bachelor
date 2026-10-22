# Introductory lecture series: Bachelor of Artificial Intelligence and Robotics Engineering

One 30-minute narrated introductory lecture per subject (49 subjects: AI.101-AI.411 and AI.450).

- `catalog/courses.csv`: the full subject list with year, semester and credit hours.
- `courses/<CODE>/course.json`: the slide script. Each slide has a layout, content and narration.
  Narration is split into clips of at most 1,500 characters (`narration` plus optional `narration_extra`).
- `courses/<CODE>/assets/`: cinematic images used on slides.
- `courses/<CODE>/audio/`: narration clips `sNN.mp3` (and `sNN_2.mp3` for extra segments). Git-ignored.
- `pipeline/build_lecture.py`: renders slides (navy and gold design, author footer on every slide),
  diagrams, the narrated MP4 with chapter markers, and a PPTX deck with narration in the speaker notes.
  Outputs go to `courses/<CODE>/output/` (git-ignored).

## Build a course

```
python lectures/pipeline/build_lecture.py lectures/courses/AI101
```

Requires Python packages `pillow`, `numpy`, `matplotlib`, `imageio-ffmpeg`, `python-pptx`.
Narration clips must exist first. Use `--allow-missing-audio --only 1-3` for a silent preview.

## Notes

- Every slide carries the footer: "TRIUNE BIBLICAL UNIVERSITY GLOBAL EXTENSION, USA".
- Voice: a single American English male voice (en-US, educational) is used for all courses.
- edge-tts was evaluated but its Microsoft speech endpoint is not reachable from the build environment,
  so narration is produced with the platform's generate_speech voice instead.
