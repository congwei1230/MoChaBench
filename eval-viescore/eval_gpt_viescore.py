import os
import json
import cv2
import base64
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from openai import OpenAI
import time

NUM_FRAMES = 32
OPENAI_API_KEY = "YOUR API KEY"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

results_dir = r"example_results"
prompt_dir = r"."
text_prompt_path = os.path.join(results_dir, "text_prompt.csv")

prompt_files = {
    "visual_quality": "visual_quality.txt",
    "facial_expression_naturalness": "facial_expression_naturalness.txt",
    "action_naturalness": "action_naturalness.txt",
    "text_alignment": "text_alignment.txt"
}

video_models = ["SadTalker", "hallo3", "MoCha", "aniportrait-square"]
video_ids = [f"{i:02}.mp4" for i in range(1, 21)]

# Load all 20 text prompts (1:1 with video_ids)
text_prompts = pd.read_csv(text_prompt_path, header=None)[0].tolist()

def encode_image_from_array(image_array):
    _, buffer = cv2.imencode('.jpg', image_array)
    return base64.b64encode(buffer).decode("utf-8")

def extract_frames(video_path, num_frames=NUM_FRAMES):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        raise ValueError(f"Video has only {total_frames} frames, but {num_frames} were requested.")
    frame_ids = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    frames = []
    for fid in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            continue
        frames.append(frame)
    cap.release()
    return frames

def parse_output(text):
    try:
        json_str = text.strip().split("```json")[1].split("```")[0]
        result = json.loads(json_str)
        return result.get("reasoning", ""), result.get("score", None)
    except:
        return "Failed to parse", None

def load_prompt(metric, prompt_idx=None):
    prompt_path = os.path.join(prompt_dir, prompt_files[metric])
    with open(prompt_path, 'r', encoding='utf-8') as f:
        template = f.read().strip()
    if metric == "text_alignment":
        assert prompt_idx is not None
        user_prompt = text_prompts[prompt_idx]
        return template.replace("<PROMPT>", user_prompt)
    else:
        return template

def evaluate_task(task):
    model, video_name, metric = task
    prompt_idx = int(video_name.split(".")[0]) - 1
    video_path = os.path.join(results_dir, model, video_name)
    prompt = load_prompt(metric, prompt_idx if metric == "text_alignment" else None)

    max_retries = 2
    attempt = 0

    while attempt <= max_retries:
        try:
            frames = extract_frames(video_path)
            content = [{"type": "input_text", "text": prompt}]
            for frame in frames:
                b64_img = encode_image_from_array(frame)
                content.append({
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{b64_img}"
                })

            client = OpenAI()
            response = client.responses.create(
                model="gpt-4.1",
                input=[{"role": "user", "content": content}]
            )
            reasoning, score = parse_output(response.output_text)
            break  # Success
        except Exception as e:
            reasoning, score = str(e), None
            if attempt < max_retries:
                print(f"[Retrying {model} | {video_name} | {metric}] Attempt {attempt + 1} failed: {e}")
                time.sleep(5)  # Sleep 5 seconds before retry
                attempt += 1
            else:
                print(f"[Failed {model} | {video_name} | {metric}] All attempts failed.")
                break

    print(f"[{model} | {video_name} | {metric}] Score: {score} | Reasoning: {reasoning}")
    return {
        "model": model,
        "video": video_name,
        "metric": metric,
        "score": score,
        "reasoning": reasoning
    }

if __name__ == "__main__":
    tasks = [(model, video, metric)
             for model in video_models
             for video in video_ids
             for metric in prompt_files.keys()]

    with Pool(processes=min(cpu_count(), 8)) as pool:
        results = pool.map(evaluate_task, tasks)

    df = pd.DataFrame(results)
    df.to_csv("gpt_viescore_results.csv", index=False)
    print("Saved to gpt_viescore_results.csv")
