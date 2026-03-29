"""短期记忆 content 内嵌多模态引用（Markdown），便于单字段展示图/音。"""

import re
from typing import Any, Dict, Optional


def media_markdown_suffix(
  origin: str,
  *,
  image_url: Optional[str] = None,
  image_ref: Optional[str] = None,
  audio_url: Optional[str] = None,
) -> str:
  """
  在 content 末尾追加 Markdown（不包含 base64，避免 JSON 臃肿）：
  - 公网图：![图片](https...)
  - 本地上传：![上传图片]({origin}/api/agent/session-image/{id}) 由服务端落盘后可访问
  - 语音：[语音播报](absolute_url)
  """
  parts: list[str] = []
  o = str(origin).rstrip("/")
  if image_url and str(image_url).strip():
    parts.append(f"\n\n![图片]({str(image_url).strip()})")
  if image_ref and isinstance(image_ref, str) and image_ref.startswith("/"):
    parts.append(f"\n\n![上传图片]({o}{image_ref})")
  if audio_url and str(audio_url).strip():
    a = str(audio_url).strip()
    href = o + a if a.startswith("/") else a
    parts.append(f"\n\n[语音播报]({href})")
  return "".join(parts)


def merge_extras_into_content(origin: str, content: str, extras: Optional[Dict[str, Any]]) -> str:
  """POST 手动追加消息时，将 extras 中的图/音补进 content（若 URL 已在正文中则不再追加）。"""
  if not extras:
    return content or ""
  c = content or ""
  suf = media_markdown_suffix(
    origin,
    image_url=extras.get("image_url") if isinstance(extras.get("image_url"), str) else None,
    image_ref=extras.get("image_ref") if isinstance(extras.get("image_ref"), str) else None,
    audio_url=extras.get("audio_url") if isinstance(extras.get("audio_url"), str) else None,
  )
  if not suf:
    return c
  urls = re.findall(r"\((https?://[^)]+|/[^)]+)\)", suf)
  if urls and all(u in c for u in urls):
    return c
  return c + suf
