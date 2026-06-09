"""
Инференс в реальном (или почти реальном) времени.

Два режима использования:
  1) Офлайн-демо (основной — для курсовой):
       demo = OfflineDemo(model, tokenizer, device)
       result = demo.run(video_path)
       # result содержит транскрипцию, WER, RTF

  2) Реальное время через веб-камеру + микрофон (bonus):
       demo = RealtimeAVSR(model, tokenizer, device)
       demo.start()   # запускает потоки захвата
       demo.stop()

Gradio-интерфейс для офлайн-демо:
       python -m src.inference.realtime --checkpoint checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from torch.cuda.amp import autocast

from src.data.preprocessing import LipROIExtractor, waveform_to_mel, SAMPLE_RATE
from src.training.metrics import decode_batch, wer as calc_wer

log = logging.getLogger("avsr.inference")


# =============================================================================
# Утилиты: читаем видеофайл целиком
# =============================================================================

def load_audio_from_video(video_path: str | Path) -> Tuple[torch.Tensor, int]:
    """
    Читает аудио из видеофайла через soundfile (если есть .wav рядом)
    или через librosa.

    Returns:
        (waveform: Tensor(T,), sample_rate: int)
    """
    video_path = Path(video_path)
    wav_path = video_path.with_suffix(".wav")

    if wav_path.exists():
        import soundfile as sf
        wav, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        return torch.from_numpy(np.ascontiguousarray(wav)), sr

    # Fallback: читаем через librosa (умеет извлекать аудио из mp4)
    try:
        import librosa
        wav, sr = librosa.load(str(video_path), sr=SAMPLE_RATE, mono=True)
        return torch.from_numpy(wav), sr
    except Exception as e:
        raise RuntimeError(
            f"Не могу прочитать аудио из {video_path}: {e}. "
            f"Конвертируйте аудио вручную: "
            f"ffmpeg -i {video_path} {wav_path}"
        )


def extract_lips_from_video(video_path: str | Path) -> torch.Tensor:
    """
    Извлекает кропы губ из видеофайла.

    Returns:
        Tensor(T_v, 1, 96, 96)
    """
    from src.data.preprocessing import video_to_lip_tensor
    return video_to_lip_tensor(video_path)


# =============================================================================
# Офлайн-демо: прогон целого видеофайла
# =============================================================================

class OfflineDemo:
    """
    Офлайн инференс: принимает путь к видео, возвращает транскрипцию и метрики.

    Используется для демонстрации в курсовой:
      - Показывает работу модели на реальных видео
      - Замеряет Real Time Factor (RTF = время инференса / длительность видео)
      - Опционально считает WER если передать reference text

    Пример:
        demo = OfflineDemo.from_checkpoint("checkpoints/best.pt")
        result = demo.transcribe("sample.mp4")
        print(result["text"], "RTF:", result["rtf"])
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        device: torch.device,
        use_amp: bool = True,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.use_amp = use_amp and device.type == "cuda"
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        config_path: Optional[str | Path] = None,
        device: Optional[torch.device] = None,
    ) -> "OfflineDemo":
        """
        Создаёт OfflineDemo из чекпоинта.
        """
        from src.utils.config import load_config
        from src.data.dataset import CharTokenizer
        from src.models.avsr_model import build_model

        checkpoint_path = Path(checkpoint_path)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config_path = config_path or checkpoint_path.parent / "config.yaml"
        cfg = load_config(config_path)
        tokenizer = CharTokenizer()
        model = build_model(cfg, vocab_size=tokenizer.vocab_size)

        ckpt = torch.load(str(checkpoint_path), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)
        model.eval()

        log.info("Модель загружена: %s -> %s", checkpoint_path.name, device)
        return cls(model, tokenizer, device)

    @torch.no_grad()
    def transcribe(
        self,
        video_path: str | Path,
        reference_text: Optional[str] = None,
    ) -> dict:
        """
        Транскрибирует видеофайл.

        Args:
            video_path:      путь к .mp4 или .avi файлу.
            reference_text:  если задан, считаем WER.

        Returns:
            dict с ключами:
              text:       str — предсказанная транскрипция
              duration:   float — длительность видео (сек)
              rtf:        float — Real Time Factor (< 1.0 = быстрее реального времени)
              wer:        float (только если передан reference_text)
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        log.info("Транскрибирую: %s", video_path.name)
        t_start = time.perf_counter()

        # --- Аудио ---
        wav, sr = load_audio_from_video(video_path)
        audio_duration = len(wav) / sr
        mel = waveform_to_mel(wav, sample_rate=sr)  # (80, T_mel)

        # --- Видео ---
        lips = extract_lips_from_video(video_path)  # (T_v, 1, 96, 96)

        t_preprocess = time.perf_counter()

        # --- Инференс ---
        # Добавляем batch-размерность
        mel_batch = mel.unsqueeze(0).to(self.device)                # (1, 80, T_mel)
        mel_lens = torch.tensor([mel.shape[1]], device=self.device) # (1,)
        video_batch = lips.unsqueeze(0).to(self.device)             # (1, T_v, 1, 96, 96)
        video_lens = torch.tensor([lips.shape[0]], device=self.device) # (1,)

        with autocast(enabled=self.use_amp):
            logits, out_lens = self.model(mel_batch, mel_lens, video_batch, video_lens)

        texts = decode_batch(logits.float(), out_lens, self.tokenizer)
        pred_text = texts[0] if texts else ""

        t_end = time.perf_counter()

        inference_time = t_end - t_start
        rtf = inference_time / max(audio_duration, 1e-6)

        result = {
            "text": pred_text,
            "duration": audio_duration,
            "inference_time": inference_time,
            "rtf": rtf,
            "preprocess_time": t_preprocess - t_start,
        }

        if reference_text is not None:
            ref_norm = self.tokenizer.normalize(reference_text)
            result["wer"] = calc_wer([pred_text], [ref_norm])
            result["reference"] = ref_norm

        log.info(
            "  Результат: '%s'  (RTF=%.3f, %.2f сек на %.2f сек видео)",
            pred_text, rtf, inference_time, audio_duration,
        )
        return result


# =============================================================================
# Gradio-интерфейс (офлайн-демо в браузере)
# =============================================================================

def launch_gradio_demo(
    checkpoint_path: str | Path,
    config_path: Optional[str | Path] = None,
    server_port: int = 7860,
    share: bool = False,
) -> None:
    """
    Запускает Gradio веб-интерфейс для демонстрации модели.

    Пользователь загружает видео → модель выдаёт транскрипцию + RTF.
    """
    try:
        import gradio as gr
    except ImportError:
        print("Установите gradio: pip install gradio")
        return

    demo_engine = OfflineDemo.from_checkpoint(checkpoint_path, config_path)

    def predict(video_file, reference_text):
        if video_file is None:
            return "Загрузите видеофайл", "", ""
        try:
            result = demo_engine.transcribe(
                video_file,
                reference_text=reference_text if reference_text else None,
            )
            transcription = result["text"]
            rtf_info = f"RTF: {result['rtf']:.3f}  |  Инференс: {result['inference_time']:.2f}с  |  Видео: {result['duration']:.2f}с"
            wer_info = f"WER: {result['wer']:.4f} ({result['wer']*100:.1f}%)" if "wer" in result else "WER: N/A (не задан reference)"
            return transcription, rtf_info, wer_info
        except Exception as e:
            return f"Ошибка: {e}", "", ""

    interface = gr.Interface(
        fn=predict,
        inputs=[
            gr.Video(label="Загрузите видео (mp4/avi)"),
            gr.Textbox(label="Эталонная транскрипция (необязательно, для WER)", placeholder="hello world"),
        ],
        outputs=[
            gr.Textbox(label="Транскрипция модели"),
            gr.Textbox(label="Скорость"),
            gr.Textbox(label="WER"),
        ],
        title="AVSR Demo — Audio-Visual Speech Recognition",
        description=(
            "Мультимодальное распознавание речи: "
            "использует одновременно аудио и движения губ."
        ),
        examples=[],
    )

    log.info("Запускаю Gradio на порту %d (share=%s)", server_port, share)
    interface.launch(server_port=server_port, share=share)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="AVSR Inference Demo")
    parser.add_argument(
        "--checkpoint", required=True,
        help="Путь к .pt чекпоинту"
    )
    parser.add_argument(
        "--config", default=None,
        help="Путь к конфигу (по умолчанию ищем рядом с чекпоинтом)"
    )
    parser.add_argument(
        "--video", default=None,
        help="Путь к видеофайлу для офлайн-транскрипции"
    )
    parser.add_argument(
        "--reference", default=None,
        help="Эталонный текст для WER"
    )
    parser.add_argument(
        "--gradio", action="store_true",
        help="Запустить Gradio веб-интерфейс"
    )
    parser.add_argument(
        "--port", type=int, default=7860,
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Создать публичную ссылку Gradio (нужен интернет)"
    )
    args = parser.parse_args()

    if args.gradio:
        launch_gradio_demo(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            server_port=args.port,
            share=args.share,
        )
    elif args.video:
        demo = OfflineDemo.from_checkpoint(args.checkpoint, args.config)
        result = demo.transcribe(args.video, reference_text=args.reference)
        print("\n--- Результат ---")
        print(f"Транскрипция: {result['text']}")
        print(f"Длительность: {result['duration']:.2f} сек")
        print(f"RTF:          {result['rtf']:.4f}")
        if "wer" in result:
            print(f"WER:          {result['wer']:.4f} ({result['wer']*100:.1f}%)")
            print(f"Reference:    {result['reference']}")
    else:
        parser.print_help()
