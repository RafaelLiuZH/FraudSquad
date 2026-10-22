# export HF_ENDPOINT=https://hf-mirror.com
# Uncomment the line above if you cannot access huggingface directly

# BERT
export HF_ENDPOINT=https://hf-mirror.com && python download.py

# Chinese embeddings
# huggingface-cli download --resume-download lier007/xiaobu-embedding-v2 --local-dir models/xiaobu-embedding-v2
# 326M
# huggingface-cli download --resume-download TencentBAC/Conan-embedding-v1  --local-dir models/Conan-embedding-v1
# 326M

# English embeddings
# huggingface-cli download --resume-download dunzhang/stella_en_400M_v5  --local-dir models/stella_en_400M_v5
# 435M parameters F32
# huggingface-cli download --resume-download dunzhang/stella_en_1.5B_v5  --local-dir models/stella_en_1.5B_v5
# 1543M f32


# Bert-base-uncased 110M, Bert-large-uncased 340M
