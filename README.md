
使用WeTextProcessing
text = "您好，欢迎致电合力亿捷，您在深圳市福田区人民法院的（2026）粤0307民初2394号知识产权纠纷案件，已依法向您预留的邮箱982145926@qq.com送达相关法律文书。您的身份证号后四位是1301。法院位置是朝南出发，经过银行后步行100米。价格是1988。您看还有什么问题？欢迎和我沟通，我随时为您服务，再见"
zh_tn_model.normalize(text)
output:
```text
您好,欢迎致电合力亿捷,您在深圳市福田区人民法院的(两千零二十六)粤零三零七民初两千三百九十四号知识产权纠纷案件,已依法向您预留的邮箱九八二幺四五九二六@qq.com送达相关法律文书.您的身份证号后四位是一千三百零一.法院位置是朝南出发,经过银行后步行一百米.价格是一千九百八十八.您看还有什么问题?欢迎和我沟通,我随时为您服务,再见
```
其中“人民法院的（2026）”转换成了“人民法院的(两千零二十六)”能不能通过自定义规则来实现”转换成“人民法院的(二零二六)”？



docker run -it --rm --gpus all -v /home/dev/xql/voxcpm2-infer:/infer \
-p 8010:8010 -p 8011:8011 \
-v /home/dev/xql/voxcpm2-lora/pretrained_models:/pretrained_models \
ghcr.io/qiliang/voxcpm2-infer:0.1.5 bash


# 修改输出
cp /infer/vllm_omni/entrypoints/openai/serving_speech.py /app/vllm-omni/vllm_omni/entrypoints/openai/serving_speech.py && \
cd /app/vllm-omni && uv pip install --python "$(python3 -c 'import sys; print(sys.executable)')" --no-cache-dir "."

export REF_AUDIO_MAX_DURATION=600.0

# 上传保存音色
curl --request POST \
  --url http://172.16.52.65:8010/v1/audio/voices \
  --header 'Authorization: Bearer sk-empty' \
  --header 'Content-Type: multipart/form-data' \
  --form name=chengna \
  --form consent=chengna \
  --form 'ref_text=嗯，可以，我们这边客服平台是一个基于网页版的系统功能的话，包括在线通话工单。' \
  --form 'audio_sample=@/Users/xiaoql/Downloads/合力咨询/合力咨询_程娜_clip9.wav'


### serve voxcmp2
vllm serve \
    /pretrained_models/VoxCPM2-lora012-merged/ \
    --served-model-name VoxCPM2 \
    --host 0.0.0.0 \
    --port 8010 \
    --omni \
    --stage-configs-path /infer/voxcpm2.yaml \
    --trust-remote-code \
    --seed 42
### serve qwen-tts
vllm serve \
    /pretrained_models/qwen_tts_chengna_1/ \
    --served-model-name QwenTTS \
    --host 0.0.0.0 \
    --port 8010 \
    --omni \
    --deploy-config /infer/qwen3_tts.yaml \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --seed 42

vllm serve \
    /pretrained_models/Qwen3-TTS-12Hz-0.6B-Base/ \
    --served-model-name QwenTTS \
    --host 0.0.0.0 \
    --port 8010 \
    --omni \
    --deploy-config /infer/qwen3_tts.yaml \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --seed 42



python end2end.py --model /pretrained_models/VoxCPM2 \
--text "您好，欢迎致电合力亿捷，您在深圳市福田区人民法院的（2026）粤0307民初2394号知识产权纠纷案件，已依法向您预留的邮箱982145926@qq.com送达相关法律文书。您的身份证号后四位是1301。法院位置是朝南出发，经过银行后步行100米。价格是1988。您看还有什么问题？欢迎和我沟通，我随时为您服务，再见" \
--output-dir /infer/output_audio


复刻声音
vllm-omni.tools.create_voice_profile \
    --model /pretrained_models/VoxCPM2 \
    --audio 合力咨询_赵娜.mp3 \
    --text "$(cat 合力咨询_赵娜.txt)" \
    --output_name "vip_voice_001"


您好，欢迎致电合力亿捷，您在深圳市福田区人民法院的（2026）粤0307民初2394号知识产权纠纷案件，已依法向您预留的邮箱982145926@qq.com送达相关法律文书。您的身份证号后四位是1301。法院位置是朝南出发，经过银行后步行100米。价格是1988元。您看还有什么问题？欢迎和我沟通，我随时为您服务，再见



python gradio_smart_voice.py --port 8011