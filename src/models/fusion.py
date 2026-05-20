"""
Модули слияния модальностей.

Реализуем ДВА варианта, чтобы потом сравнить (это будет важная аблация в
курсовой):

1) ConcatFusion — простое решение:
   - интерполируем видео-фичи до длины аудио (или наоборот);
   - конкатенируем по фичевой оси: (B, T, d_a + d_v);
   - линейная проекция обратно в d_model.

2) CrossAttentionFusion — основной вариант:
   - аудио-фичи (Q) делают attention к видео-фичам (K, V)  → аудио, обогащённое видео;
   - видео-фичи (Q) делают attention к аудио-фичам (K, V)  → видео, обогащённое аудио;
   - 2 таких блока подряд, в конце — конкатенация и проекция.

Зачем оба: показать в эксперименте, что cross-attention лучше при шуме.
"""
from __future__ import annotations

# TODO(day-10): импорты — torch, torch.nn, torch.nn.functional

# TODO(day-10): class ConcatFusion(nn.Module)
#   __init__(d_audio, d_video, d_model)
#   forward(audio: (B, Ta, d_a), audio_lens,
#           video: (B, Tv, d_v), video_lens)
#     — F.interpolate(video.transpose(1,2), size=Ta).transpose(1,2)
#     — concat по dim=-1 -> Linear -> (B, Ta, d_model)
#   возвращаем (fused, audio_lens)   # длину наследуем от аудио

# TODO(day-10): class CrossAttentionFusion(nn.Module)
#   __init__(d_model, n_heads=8, n_layers=2, dropout=0.1)
#   forward(audio, audio_lens, video, video_lens) -> (fused, lens)
#     — 2 блока:
#         a' = a + MHA(q=a, k=v, v=v); a' = a' + FFN(a')
#         v' = v + MHA(q=v, k=a, v=a); v' = v' + FFN(v')
#     — финал: интерполировать v' до Ta, concat, Linear
