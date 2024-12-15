from huggingface_hub import snapshot_download
local_dir = "/projectnb/vkolagrp/ketanss/scope-diffusers/sdpcache"
snapshot_download(repo_id="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",local_dir=local_dir,repo_type="model")