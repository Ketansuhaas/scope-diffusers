from huggingface_hub import snapshot_download
local_dir = "/projectnb/vkolagrp/ketanss/scope-diffusers/sdpcache"
snapshot_download(repo_id="openai/clip-vit-large-patch14",local_dir=local_dir,repo_type="model")