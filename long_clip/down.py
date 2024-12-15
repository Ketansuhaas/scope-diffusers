from huggingface_hub import snapshot_download
local_dir = "/projectnb/ivc-ml/xthomas/cs791/Long-CLIP/checkpoints/long_clip_b"
snapshot_download(repo_id="BeichenZhang/LongCLIP-B",local_dir=local_dir,repo_type="model")