# --- Libraries ---
# Streamlit for UI rendering
import streamlit as st
# OpenAI Whisper for Japanese speech-to-text
import whisper
# YouTube downloader
import yt_dlp
# OS and file handling
import os
# Video manipulation
from moviepy.video.io.VideoFileClip import VideoFileClip
# Image processing
import cv2
import numpy as np
import re
import shutil
import stat
import pandas as pd
import tempfile

# --- Japanese sentiment keywords ---
# Positive and negative words used to score emotional intensity in speech
POSITIVE_WORDS = ["楽しい", "嬉しい", "素晴らしい", "最高", "好き", "感動", "笑", "幸せ"]
NEGATIVE_WORDS = ["悲しい", "嫌い", "つらい", "苦しい", "痛い", "怖い", "怒り"]

# --- Streamlit UI setup ---
st.title("🎥 Japanese Video Highlight Extractor with Hook Score")
video_url = st.text_input("Paste YouTube video URL")
segment_duration = st.slider("Highlight Duration (seconds)", min_value=5, max_value=30, value=20)

# Main processing starts when button is clicked
if st.button("Start Processing") and video_url:

    # --- Extract YouTube video ID from URL ---
    def extract_video_id(url):
        m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
        return m.group(1) if m else None

    video_id = extract_video_id(video_url)
    video_filename = f"{video_id}.mp4"
    output_dir = f"clips_{video_id}"

    # --- Cleanup old videos that are not current ---
    for fname in os.listdir():
        if fname.endswith(".mp4") and video_id not in fname:
            try:
                os.remove(fname)
            except Exception as e:
                st.warning(f"⚠️ Could not delete {fname}: {e}")

    # Helper function to remove read-only files when deleting folder
    def remove_readonly(func, path, _):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    # Remove old output folder if exists
    if os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir, onerror=remove_readonly)
        except Exception as e:
            st.warning(f"⚠️ Could not delete output folder: {e}")

    os.makedirs(output_dir, exist_ok=True)
    st.write("🔍 Video ID:", video_id)

    # --- Download YouTube video ---
    def download_video(url, out):
        ydl_opts = {
            'format': 'bestvideo+bestaudio/best',  # Highest quality
            'outtmpl': out,  # Output file path
            'merge_output_format': 'mp4',
            'quiet': True
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            return True
        except Exception as e:
            st.error(f"❌ Download failed: {e}")
            return False

    with st.spinner("📥 Downloading video..."):
        ok = download_video(video_url, video_filename)

    if not ok or not os.path.isfile(video_filename):
        st.stop()

    # --- Transcribe Japanese speech using Whisper ---
    with st.spinner("📝 Transcribing audio (Whisper base)..."):
        model = whisper.load_model("base")
        result = model.transcribe(video_filename, language="ja")
        segments = result.get("segments", [])

    # --- Text scoring function based on sentiment words ---
    def score_text(text):
        positive_count = sum(text.count(word) for word in POSITIVE_WORDS)
        negative_count = sum(text.count(word) for word in NEGATIVE_WORDS)
        word_count = len(text.split())

        sentiment = 5 + (2 * positive_count) - (2 * negative_count)  # Base score
        sentiment = max(1, min(10, sentiment))  # Clamp between 1–10

        length_bonus = min(2.0, word_count / 10.0)  # Slight boost for longer text
        boosted_score = sentiment + length_bonus
        return min(10, boosted_score)

    # --- Image scoring function based on brightness & contrast ---
    def score_image(img_path):
        try:
            img = cv2.imread(img_path)
            if img is None:
                return 5.0
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)       # Average pixel value
            contrast = gray.std()            # Standard deviation of pixel values
            bright_score = brightness / 25.5
            contrast_score = contrast / 12.75
            return min(10, (0.4 * bright_score + 0.6 * contrast_score) * 2)
        except:
            return 5.0

    # --- Clean filler words & symbols from transcription ---
    def clean_text(text):
        text = re.sub(r"[「」『』（）()]", "", text)
        text = re.sub(r"(えー|あー|うーん)", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # --- Analyze video segments and assign scores ---
    def analyze_segments(filename, segs, max_duration=150):
        clip = VideoFileClip(filename)
        if clip.duration > max_duration:
            clip = clip.subclip(0, max_duration)  # Limit processing for speed

        data = []
        for s in segs:
            stime, etime = int(s["start"]), int(s["end"])
            if stime >= max_duration or (etime - stime < 5):  # Skip very short segments
                continue

            clean = clean_text(s["text"])
            text_score = score_text(clean)

            # --- Hook score: first 1–2 seconds of segment ---
            hook_scores = []
            hook_end_time = min(stime + 2, etime)
            for t in np.linspace(stime, hook_end_time, num=2):
                fpath = os.path.join(output_dir, f"hook_frame_{t:.1f}.jpg")
                try:
                    clip.save_frame(fpath, t)
                    hook_scores.append(score_image(fpath))
                except:
                    continue
            hook_score = np.mean(hook_scores) if hook_scores else 5.0

            # --- Image score for entire segment ---
            frame_scores = []
            for t in range(stime, min(etime, int(clip.duration)), 3):
                fpath = os.path.join(output_dir, f"frame_{t}.jpg")
                try:
                    clip.save_frame(fpath, t)
                    frame_scores.append(score_image(fpath))
                except:
                    continue

            if not frame_scores:
                continue

            avg_frame = sum(frame_scores) / len(frame_scores)

            # --- Weighted total score calculation ---
            weighted_score = (0.6 * text_score) + (0.25 * avg_frame) + (0.15 * hook_score)
            if text_score > 6 and avg_frame > 6:
                weighted_score += 0.5  # Bonus for strong text+image match

            data.append({
                "start": stime,
                "end": etime,
                "text": clean,
                "text_score": text_score,
                "image_score": avg_frame,
                "hook_score": hook_score,
                "total_score": min(10, weighted_score)
            })

        return pd.DataFrame(data)

    # --- Normalize total scores to 6–10 range ---
    def normalize_scores(df):
        min_score = df["total_score"].min()
        max_score = df["total_score"].max()
        if max_score - min_score < 0.1:  # Avoid divide by zero
            return df
        df["total_score"] = ((df["total_score"] - min_score) / (max_score - min_score)) * 4 + 6
        return df

    # --- Main scoring process ---
    with st.spinner("🔍 Scoring segments..."):
        df_scores = analyze_segments(video_filename, segments, max_duration=1500)
        df_scores = normalize_scores(df_scores)

    # Display scores & chart
    st.subheader("📊 Emotional + Hook Score Progression")
    st.dataframe(df_scores)
    st.line_chart(df_scores.set_index("start")[["total_score", "hook_score"]])
    st.download_button("⬇️ Download Scores as CSV", df_scores.to_csv(index=False), file_name="emotion_hook_scores.csv")

    # --- Select top 3 clips ---
    top_clips = df_scores.sort_values("total_score", ascending=False).head(3).to_dict(orient="records")

    # --- Generate and show highlight videos ---
    with st.spinner("🎞️ Generating highlights..."):
        clip = VideoFileClip(video_filename)
        for i, best in enumerate(top_clips):
            subclip = clip.subclip(best["start"], best["start"] + segment_duration)
            highlight_path = os.path.join(output_dir, f"highlight_{i+1}.mp4")
            subclip.write_videofile(highlight_path, logger=None)
            st.video(highlight_path)

            # Show clip metadata
            st.write(f"🎬 Highlight {i+1} (Score: {best['total_score']:.1f})")
            st.write(f"⏱️ {best['start']}s - {best['start'] + segment_duration}s")
            st.write(f"💬 {best['text']}")

    st.success("✅ Done!")
