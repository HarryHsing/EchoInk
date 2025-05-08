import base64
from io import BytesIO
import audioread
import av
import librosa
import numpy as np

# 默认最大音频长度，但现在可以在调用时覆盖
DEFAULT_MAX_AUDIO_LEN_SEC = 10

def _check_if_video_has_audio(video_path):
    container = av.open(video_path)
    audio_streams = [stream for stream in container.streams if stream.type == "audio"]
    return bool(audio_streams)

def _load_audio_with_limit(path, max_seconds=None):
    """统一方式加载并裁剪音频
    
    Args:
        path: 音频路径
        max_seconds: 最大音频长度(秒)，None表示不限制
    """
    if max_seconds is not None:
        # 使用librosa的duration参数直接限制加载时间
        audio, _ = librosa.load(path, sr=16000, duration=max_seconds)
        return audio
    else:
        # 不限制，加载完整音频
        audio, _ = librosa.load(path, sr=16000)
        return audio

def process_audio_info(
    conversations: list[dict] | list[list[dict]], 
    use_audio_in_video: bool = True,
    max_audio_seconds: float = DEFAULT_MAX_AUDIO_LEN_SEC,  # 参数化最大音频长度
    enable_audio_processing: bool = True  # 控制是否处理音频
):
    """
    处理对话中的音频信息
    
    Args:
        conversations: 对话数据
        use_audio_in_video: 是否处理视频中的音频
        max_audio_seconds: 最大音频长度(秒)，None表示不限制
        enable_audio_processing: 是否启用音频处理
    """
    if not enable_audio_processing:
        return None
        
    audios = []
    max_samples = None if max_audio_seconds is None else int(16000 * max_audio_seconds)
    
    if isinstance(conversations[0], dict):
        conversations = [conversations]

    for conversation in conversations:
        for message in conversation:
            if not isinstance(message["content"], list):
                continue
            for ele in message["content"]:
                # if (ele["type"] == "audio" or ele["type"] == "image_audio") and "audio" in ele:
                if ele["type"] == "audio" and "audio" in ele:
                    path = ele["audio"]
                    if isinstance(path, np.ndarray):
                        if path.ndim > 1:
                            raise ValueError("Support only mono audio")
                        # 只有在设置了max_samples时才截断
                        audio_data = path[:max_samples] if max_samples is not None else path
                        audios.append(audio_data)
                    elif path.startswith("data:audio"):
                        _, base64_data = path.split("base64,", 1)
                        data = base64.b64decode(base64_data)
                        audio = librosa.load(BytesIO(data), sr=16000, duration=max_audio_seconds)[0]
                        audios.append(audio)
                    elif path.startswith("http://") or path.startswith("https://"):
                        audio = librosa.load(audioread.ffdec.FFmpegAudioFile(path), sr=16000, duration=max_audio_seconds)[0]
                        audios.append(audio)
                    elif path.startswith("file://"):
                        audios.append(_load_audio_with_limit(path[len("file://"):], max_audio_seconds))
                    else:
                        audios.append(_load_audio_with_limit(path, max_audio_seconds))

                if use_audio_in_video and ele["type"] == "video" and "video" in ele:
                    path = ele["video"]
                    assert _check_if_video_has_audio(path), \
                        "Video must have audio track when use_audio_in_video=True"
                    if path.startswith("http://") or path.startswith("https://"):
                        audio = librosa.load(audioread.ffdec.FFmpegAudioFile(path), sr=16000, duration=max_audio_seconds)[0]
                        audios.append(audio)
                    elif path.startswith("file://"):
                        audios.append(_load_audio_with_limit(path[len("file://"):], max_audio_seconds))
                    else:
                        audios.append(_load_audio_with_limit(path, max_audio_seconds))

    return audios if audios else None
