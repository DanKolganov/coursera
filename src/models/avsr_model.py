"""
Главная модель AVSR — собирает все компоненты воедино.

Архитектура:
  audio_mel ──► AudioEncoder ──┐
                                ├──► Fusion ──► CTC Head ──► logits
  lip_video ──► VideoEncoder ──┘

Поддерживает три режима:
  - "audio_only":  игнорирует видео (для бейзлайна)
  - "video_only":  игнорирует аудио (бейзлайн чтения по губам)
  - "av":          мультимодальный (основной)

Modality dropout (важная штука): при обучении с вероятностью p_drop
случайно зануляем одну из веток. Это учит модель не разваливаться,
если в инференсе одна модальность пропала (например, лицо не нашлось).
"""
from __future__ import annotations

# TODO(day-10): импорты — torch, наши модули

# TODO(day-10): class AVSRModel(nn.Module)
#   __init__(cfg):
#     self.audio_encoder = AudioEncoder(...)
#     self.video_encoder = VideoEncoder(...)
#     self.fusion = CrossAttentionFusion(...) or ConcatFusion(...)
#     self.ctc_head = nn.Linear(d_model, vocab_size)
#     self.mode = cfg.model.mode  # "av" / "audio_only" / "video_only"
#     self.modality_dropout = cfg.model.modality_dropout
#
#   forward(audio_mel, audio_lens, video, video_lens) -> (logits, out_lens):
#     a, a_lens = self.audio_encoder(audio_mel, audio_lens)
#     v, v_lens = self.video_encoder(video, video_lens)
#     # modality dropout (только в training)
#     if self.training and self.modality_dropout > 0:
#         r = torch.rand(1).item()
#         if r < self.modality_dropout: a = torch.zeros_like(a)
#         elif r < 2*self.modality_dropout: v = torch.zeros_like(v)
#     fused, lens = self.fusion(a, a_lens, v, v_lens)
#     logits = self.ctc_head(fused)            # (B, T, vocab)
#     return logits, lens
