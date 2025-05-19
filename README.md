# MoChaBench


## 🏆 Leaderboard

### Per-Category Averages

| Category                      | Sync-Dist | Sync-Conf | Count (n) |
|------------------------------|----------|------------|-----------|
| 1p_camera_movement           | 8.455    | 5.432      | 18        |
| 1p_closeup_facingcamera      | 7.958    | 6.298      | 27        |
| 1p_emotion                   | 8.073    | 6.214      | 34        |
| 1p_generalize_chinese        | 8.273    | 4.398      | 4         |
| 1p_mediumshot_actioncontrol  | 8.386    | 6.241      | 52        |
| 1p_protrait                  | 8.125    | 6.892      | 38        |
| 2p_1clip_1talk               | 8.082    | 6.493      | 30        |
| 2p_2clip_2talk               | 8.601    | 4.951      | 15        |

### Aggregate Group Average
| Group                            | Distance | Confidence |
|----------------------------------|----------|------------|
| Single-character English Monologue (1p_camera_movement + 1p_closeup_facingcamera + 1p_emotion + 1p_mediumshot_actioncontrol + 1p_protrait + 2p_1clip_1talk)  | 8.185    | 6.333      |
| Turn-based English Dialogue (2p_2clip_2talk)  | 8.601    | 4.951      |



## Download this repo
Benchmark and MoChaGeneration Results are embedded in this git repo
```
cd local_repo_dir
git clone https://github.com/congwei1230/MoChaBench.git
```

## Evaluate Lip Sync Scores with SyncNet:

### Download this repo
Model weights are embedded in this git repo
```
git clone https://github.com/congwei1230/MoChaBench.git
```

### Dependencies
```
conda create -n mochabench_eval python=3.8
conda activate mochabench_eval
pip install -r requirements.txt
# require ffmpeg installed
```

### Overview
This script is adapted from [joonson/syncnet_python](https://github.com/joonson/syncnet_python) for improved API and code structure.

Follows a HuggingFace Diffuser-style structure.
We provided a
`SyncNetPipeline` Class located at `eval-lipsync\script\syncnet_pipeline.py`
`SyncNetPipeline` can be intialized by providing the weights and configs.

```
pipe = SyncNetPipeline(
    {
        "s3fd_weights":  "../weights/sfd_face.pth",
        "syncnet_weights": "../weights/syncnet_v2.model",
    },
    device="cuda",          # or "cpu"
)
```
It has a inference function to score a single pair of video and speech(denoised from audio)
```
results = pipe.inference(
    video_path="../example/video.avi",   # RGB video
    audio_path="../example/speech.wav",   # speech track (any ffmpeg-readable format)
    cache_dir="../example/cache",    # optional; omit to auto-cleanup intermediates
)
```

### Example Script to run SyncNetPipeline on single pair of (video, speech)

```
cd eval-lipsync\script
python run_syncnet_pipeline_on_1example.py
```
You are expected to get
```
AV offset:      1
Min dist:       9.255
Confidence:     4.497
best-confidence   : 4.4973907470703125
lowest distance   : 9.255396842956543
per-crop offsets  : [1]
```

### Script to run SyncNetPipeline on MoCha Generation Results on MoChaBench

```
cd eval-lipsync\script
python run_syncnet_pipeline_on_mocha_generation_on_mocha_bench.py
```

### Script to run SyncNetPipeline on Custom Models' Results on MoChaBench
You need to create a folder similar to the structure of ``local_repo_dir/remocha-generation``
Then modify the 
```
cd eval-lipsync\script
python run_syncnet_pipeline_on_mocha_generation_on_mocha_bench.py
```

