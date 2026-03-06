#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
高棉语 TTS 网页界面（带 VAD 对齐和 Whisper 微调数据生成）
- 生成完整音频（16kHz 单声道 16-bit PCM）
- 使用 VAD 检测语音段，生成对齐的双语字幕和语音片段
- 始终输出 Whisper 微调 JSON
- 容错：对生成失败的句子插入 0.1 秒静音
- 改进：合并相邻且文本相同的语音段，彻底解决字幕重复问题
"""

import os
import re
import sys
import json
import tempfile
import zipfile
import time
from pathlib import Path

import numpy as np
import gradio as gr
import torch
import requests
from transformers import VitsModel, AutoTokenizer
import webrtcvad
import librosa
import soundfile as sf
from transformers import VitsModel, AutoTokenizer

# ---------- 自动安装缺失依赖 ----------
required_packages = ["webrtcvad", "librosa", "soundfile"]
for pkg in required_packages:
    try:
        __import__(pkg)
    except ImportError:
        print(f"正在安装 {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        globals()[pkg] = __import__(pkg)

import webrtcvad
import librosa
import soundfile as sf

# ---------- 配置 ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# 加载模型
model = VitsModel.from_pretrained("facebook/mms-tts-khm").to(device)
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-khm")
model_sample_rate = model.config.sampling_rate  # 原始模型输出采样率 (22050 Hz)

# 目标采样率（Whisper 标准）
TARGET_SR = 16000

# 短句跳过阈值（字符数）
MIN_SENTENCE_LENGTH = 5
# VAD 敏感度 (0-3)
VAD_AGGRESSIVENESS = 2
# 最小语音段时长（秒）
MIN_SEGMENT_DURATION = 0.3
# 生成失败时插入的静音时长（秒）
FALLBACK_SILENCE_DURATION = 0.1

# DeepSeek API 密钥（建议从环境变量读取）
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-ef45f20cae4447f6bd26d24ef20d29df")

# ---------- 辅助函数 ----------
def format_srt_time(seconds: float) -> str:
    """将秒转换为 SRT 时间格式 HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def translate_text(text: str, max_retries=3) -> str:
    """使用 DeepSeek API 将高棉语翻译成中文，带重试机制"""
    if not text.strip() or len(text) < MIN_SENTENCE_LENGTH:
        return "[跳过翻译]"
    
    url = "https://api.deepseek.com/v1/chat/completions"
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一个专业的翻译助手，擅长高棉语到中文的翻译。"},
            {"role": "user", "content": f"请将以下高棉语翻译成中文，只返回翻译结果：\n\n{text}"}
        ],
        "temperature": 0.3,
        "max_tokens": 1000
    }
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    
    for attempt in range(max_retries):
        try:
            json_bytes = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            response = requests.post(
                url,
                headers=headers,
                data=json_bytes,
                timeout=(10, 30)  # 连接超时10秒，读取超时30秒
            )
            if response.status_code == 200:
                result = response.json()
                translated = result["choices"][0]["message"]["content"].strip().strip('"\'')
                return translated
            else:
                print(f"翻译API返回错误 {response.status_code}，重试 {attempt+1}/{max_retries}", flush=True)
                time.sleep(1 * (attempt + 1))  # 递增等待
        except requests.exceptions.Timeout:
            print(f"翻译超时，重试 {attempt+1}/{max_retries}", flush=True)
            time.sleep(1 * (attempt + 1))
        except Exception as e:
            print(f"翻译异常: {e}，重试 {attempt+1}/{max_retries}", flush=True)
            time.sleep(1 * (attempt + 1))
    
    # 所有重试失败后，返回一个占位符，但程序继续
    return f"[翻译失败] {text[:50]}..."

def split_into_clauses(sentence: str) -> list:
    """
    将句子进一步拆分为子句，按逗号分割（英文逗号和高棉语可能使用的逗号）。
    返回子句列表，保留标点（可选）。
    """
    # 分割符：逗号（英文和中文）、分号、冒号等
    parts = re.split(r'[,，;；:]', sentence)
    clauses = [p.strip() for p in parts if p.strip()]
    return clauses if clauses else [sentence]

def convert_to_whisper_format(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """
    将音频转换为 Whisper 标准格式：
    - 单声道（若为立体声则取平均）
    - 重采样至 TARGET_SR (16000 Hz)
    - 归一化并转换为 int16
    返回 int16 数组
    """
    if len(audio) == 0:
        return np.array([], dtype=np.int16)
    
    # 转单声道
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    
    # 重采样
    if orig_sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=TARGET_SR)
    
    # 去除直流偏移并归一化至 0.95 振幅
    audio = audio - np.mean(audio)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.95
    
    # 转换为 int16
    int16_audio = (audio * 32767).astype(np.int16)
    return int16_audio

def generate_full_audio(clauses):
    """
    生成完整音频（原始采样率），并对生成失败的子句插入静音。
    返回：
    - full_audio_raw: numpy 数组，原始采样率
    - clause_infos: 列表，每项为 (text, start_time, duration) 基于原始采样率
    - orig_sr: 原始采样率
    """
    audio_segments = []
    clause_infos = []
    current_time = 0.0
    total = len(clauses)
    
    for i, clause in enumerate(clauses, 1):
        print(f"生成语音第 {i}/{total} 句（子句）...", flush=True)
        
        # 检查 tokenize 后是否为空
        inputs = tokenizer(clause, return_tensors="pt").to(device)
        if inputs['input_ids'].shape[1] == 0:
            print(f"  警告: 子句 '{clause[:30]}' tokenize 后为空，插入 {FALLBACK_SILENCE_DURATION} 秒静音", flush=True)
            duration = FALLBACK_SILENCE_DURATION
            audio = np.zeros(int(model_sample_rate * duration), dtype=np.float32)
            audio_segments.append(audio)
            clause_infos.append((clause, current_time, duration))
            current_time += duration
            continue
        
        try:
            with torch.no_grad():
                output = model(**inputs).waveform
            audio = output.squeeze().cpu().numpy()
            duration = len(audio) / model_sample_rate
            audio_segments.append(audio)
            clause_infos.append((clause, current_time, duration))
            current_time += duration
        except Exception as e:
            print(f"  错误: 生成子句 '{clause[:30]}' 时发生异常: {e}，插入 {FALLBACK_SILENCE_DURATION} 秒静音", flush=True)
            duration = FALLBACK_SILENCE_DURATION
            audio = np.zeros(int(model_sample_rate * duration), dtype=np.float32)
            audio_segments.append(audio)
            clause_infos.append((clause, current_time, duration))
            current_time += duration
    
    full_audio_raw = np.concatenate(audio_segments) if audio_segments else np.array([], dtype=np.float32)
    return full_audio_raw, clause_infos, model_sample_rate

def vad_split(audio_int16: np.ndarray, sample_rate: int) -> list:
    """
    使用 VAD 检测语音段，返回 (start_sec, end_sec) 列表。
    输入 audio_int16 为 int16 数组。
    """
    if len(audio_int16) == 0:
        return []
    
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    
    # 转换为字节流
    audio_bytes = audio_int16.tobytes()
    
    frame_duration_ms = 30
    frame_size = int(sample_rate * frame_duration_ms / 1000) * 2  # 16-bit = 2 bytes
    
    is_speech = []
    for i in range(0, len(audio_bytes) - frame_size + 1, frame_size):
        frame = audio_bytes[i:i + frame_size]
        is_speech.append(vad.is_speech(frame, sample_rate))
    
    # 合并连续语音帧
    segments = []
    in_speech = False
    start_frame = 0
    
    for i, speech in enumerate(is_speech):
        if speech and not in_speech:
            in_speech = True
            start_frame = i
        elif not speech and in_speech:
            in_speech = False
            end_frame = i
            start_time = start_frame * frame_duration_ms / 1000
            end_time = end_frame * frame_duration_ms / 1000
            if end_time - start_time > MIN_SEGMENT_DURATION:
                segments.append((start_time, end_time))
    
    if in_speech:
        end_frame = len(is_speech)
        start_time = start_frame * frame_duration_ms / 1000
        end_time = end_frame * frame_duration_ms / 1000
        if end_time - start_time > MIN_SEGMENT_DURATION:
            segments.append((start_time, end_time))
    
    return segments

def match_text_to_segments(vad_segments, clause_infos):
    """
    为每个 VAD 段分配最匹配的子句文本（基于时间中心点）。
    返回与 vad_segments 等长的文本列表。
    """
    assigned = []
    for v_start, v_end in vad_segments:
        mid = (v_start + v_end) / 2
        best_idx = -1
        min_dist = float('inf')
        for i, (text, c_start, c_dur) in enumerate(clause_infos):
            c_end = c_start + c_dur
            if c_start <= mid <= c_end:
                best_idx = i
                break
            else:
                dist = min(abs(mid - c_start), abs(mid - c_end))
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
        if best_idx >= 0:
            assigned.append(clause_infos[best_idx][0])
        else:
            assigned.append("[未知]")
    return assigned

def merge_duplicate_segments(segments):
    """
    合并相邻且文本相同的段。
    segments: list of (start, end, text)
    返回合并后的列表。
    """
    if not segments:
        return segments
    merged = []
    cur_start, cur_end, cur_text = segments[0]
    for start, end, text in segments[1:]:
        if text == cur_text:
            # 合并：延长结束时间
            cur_end = end
        else:
            merged.append((cur_start, cur_end, cur_text))
            cur_start, cur_end, cur_text = start, end, text
    merged.append((cur_start, cur_end, cur_text))
    return merged

# ---------- 核心生成函数 ----------
def generate_aligned_package(text: str, custom_filename: str):
    """生成对齐的数据包，返回 ZIP 文件路径"""
    if not text.strip():
        return None

    # 处理文件名
    if not custom_filename:
        raw = re.sub(r'\s+', '', text)
        custom_filename = raw[:10] if raw else "output"
        custom_filename = re.sub(r'[\\/*?:"<>|]', "", custom_filename)

    # 分句：先按句号分割，再按逗号拆分成子句
    raw_sentences = re.split(r'[។៕?!]', text)
    clauses = []  # 最终的子句列表
    for sent in raw_sentences:
        sent = sent.strip()
        if not sent:
            continue
        # 将长句子按逗号拆分成子句
        sub_clauses = split_into_clauses(sent)
        for cl in sub_clauses:
            if len(cl) >= MIN_SENTENCE_LENGTH:
                clauses.append(cl)
            else:
                print(f"⚠ 跳过过短子句: {cl[:20]}... (长度 {len(cl)})", flush=True)
    
    total = len(clauses)
    if total == 0:
        print("错误：没有有效的子句")
        return None
    print(f"有效子句数: {total} 句", flush=True)

    # 临时工作目录
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. 生成完整音频（原始采样率）
        full_audio_raw, clause_infos, orig_sr = generate_full_audio(clauses)

        if len(full_audio_raw) == 0:
            print("错误：生成的音频为空")
            return None

        # 2. 转换为 Whisper 格式（16000 Hz, int16）
        full_audio_16k = convert_to_whisper_format(full_audio_raw, orig_sr)
        if len(full_audio_16k) == 0:
            print("错误：转换后音频为空")
            return None
        current_sr = TARGET_SR

        # 保存完整音频（无 _full 后缀）
        full_audio_path = os.path.join(tmpdir, f"{custom_filename}.wav")
        sf.write(full_audio_path, full_audio_16k, current_sr, subtype='PCM_16')
        print(f"完整音频已保存: {full_audio_path}")

        # 3. VAD 检测语音段（基于 16kHz 音频）
        vad_segments = vad_split(full_audio_16k, current_sr)
        print(f"VAD 检测到 {len(vad_segments)} 个语音段", flush=True)

        if len(vad_segments) == 0:
            print("警告：VAD 未检测到任何语音段，可能音频全是静音")
            # 仍然继续，但后续可能没有片段

        # 4. 为每个 VAD 段分配文本
        aligned_texts = match_text_to_segments(vad_segments, clause_infos)

        # 5. 构建初步对齐段 (start, end, text)
        raw_segments = [(vad_segments[i][0], vad_segments[i][1], aligned_texts[i]) for i in range(len(vad_segments))]

        # 6. 合并相邻且文本相同的段
        aligned_segments = merge_duplicate_segments(raw_segments)
        print(f"合并后剩余 {len(aligned_segments)} 个语音段", flush=True)

        # 7. 提取语音片段并保存
        segment_paths = []
        for idx, (start, end, text) in enumerate(aligned_segments):
            start_sample = int(start * current_sr)
            end_sample = int(end * current_sr)
            seg_audio = full_audio_16k[start_sample:end_sample]
            seg_path = os.path.join(tmpdir, f"{custom_filename}_seg_{idx:04d}.wav")
            sf.write(seg_path, seg_audio, current_sr, subtype='PCM_16')
            segment_paths.append(seg_path)

        # 8. 生成高棉语字幕
        km_srt_path = os.path.join(tmpdir, f"{custom_filename}.km.srt")
        with open(km_srt_path, "w", encoding="utf-8") as f:
            for i, (start, end, txt) in enumerate(aligned_segments, 1):
                f.write(f"{i}\n{format_srt_time(start)} --> {format_srt_time(end)}\n{txt}\n\n")

        # 9. 翻译为中文（逐段）
        print("开始逐段翻译...", flush=True)
        zh_segments = []
        for i, (start, end, kh_txt) in enumerate(aligned_segments):
            print(f"翻译第 {i+1}/{len(aligned_segments)} 段...", flush=True)
            zh_txt = translate_text(kh_txt)
            zh_segments.append((start, end, zh_txt))

        # 10. 生成中文字幕
        zh_srt_path = os.path.join(tmpdir, f"{custom_filename}.zh.srt")
        with open(zh_srt_path, "w", encoding="utf-8") as f:
            for i, (start, end, txt) in enumerate(zh_segments, 1):
                f.write(f"{i}\n{format_srt_time(start)} --> {format_srt_time(end)}\n{txt}\n\n")

        # 11. 生成 Whisper JSON（始终生成）
        whisper_data = []
        for idx, ((start, end, kh_txt), seg_path) in enumerate(zip(aligned_segments, segment_paths)):
            rel_path = os.path.basename(seg_path)
            whisper_data.append({
                "audio_filepath": rel_path,
                "text": kh_txt,
                "duration": end - start
            })
        json_path = os.path.join(tmpdir, f"{custom_filename}_whisper.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(whisper_data, f, ensure_ascii=False, indent=2)

        # 12. 打包 ZIP
        zip_path = os.path.join(tmpdir, f"{custom_filename}.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(full_audio_path, arcname=f"{custom_filename}.wav")
            zipf.write(km_srt_path, arcname=f"{custom_filename}.km.srt")
            zipf.write(zh_srt_path, arcname=f"{custom_filename}.zh.srt")
            for seg_path in segment_paths:
                zipf.write(seg_path, arcname=os.path.basename(seg_path))
            zipf.write(json_path, arcname=f"{custom_filename}_whisper.json")

        # 复制到持久化临时文件
        final_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        final_zip.close()
        with open(zip_path, 'rb') as f_in, open(final_zip.name, 'wb') as f_out:
            f_out.write(f_in.read())

    # 清理 GPU 缓存
    if device == "cuda":
        torch.cuda.empty_cache()

    return final_zip.name

# ---------- Gradio 界面 ----------
with gr.Blocks(title="高棉语TTS + VAD对齐 (Whisper微调数据)", theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# 🎤 高棉语文本转语音 + 精准对齐 (GPU: {device.upper()})")
    gr.Markdown("""
    - 生成符合 Whisper 微调标准的音频（单声道、16kHz、16-bit PCM）
    - 使用 VAD 检测真实语音边界，确保字幕与音频严格对齐
    - **改进**：长句子按逗号拆分为子句，并自动合并相邻且文本相同的语音段，彻底消除字幕重复
    - 自动输出：完整音频、对齐的双语字幕、每个语音段的音频、Whisper JSON 数据
    - 容错：对生成失败的句子自动插入短静音，保证流程不中断
    """)

    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="高棉语文章",
                lines=8,
                placeholder="请粘贴你的高棉语文章..."
            )
            file_name = gr.Textbox(
                label="自定义文件名（留空则自动生成）",
                placeholder="例如：my_story"
            )
            with gr.Row():
                generate_btn = gr.Button("生成并打包下载", variant="primary")
                clear_btn = gr.Button("清空")
        with gr.Column():
            output_zip = gr.File(label="下载ZIP压缩包")
            gr.Markdown("""
            ### 📦 ZIP包内含：
            - `{filename}.wav` 完整合成音频（16kHz单声道）
            - `{filename}.km.srt` 对齐的高棉语字幕
            - `{filename}.zh.srt` 对齐的中文字幕
            - `{filename}_seg_*.wav` 每个语音段单独文件
            - `{filename}_whisper.json` Whisper 微调数据
            """)

    generate_btn.click(
        fn=generate_aligned_package,
        inputs=[input_text, file_name],
        outputs=output_zip
    )

    clear_btn.click(
        fn=lambda: ("", "", None),
        inputs=[],
        outputs=[input_text, file_name, output_zip]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7899, quiet=False)