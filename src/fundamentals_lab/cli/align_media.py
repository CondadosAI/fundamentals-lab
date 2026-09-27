"""Animated figures for unit 3.5 (needs ffmpeg with libwebp)."""

import click

from fundamentals_lab.alignment import media


@click.command()
@click.option("--only", multiple=True, help="Render these animations only (by name).")
def align_media(only: tuple[str, ...]) -> None:
    """Render every practical-example animation the posts open with, as animated WebP."""
    if only:
        s = media.Scene()
        table = {
            "minimap": lambda: (media.minimap(s), 760),
            "virtual-ad": lambda: (media.virtual_ad(s), 640),
            "document-scan": lambda: (media.document_scan(), 640),
            "panorama": lambda: (media.panorama_anim(), 720),
            "calibration": lambda: (media.calibration(), 560),
            "vanishing-point": lambda: (media.vanishing_point(s), 640),
            "linear-morph": lambda: (media.linear_morph(), 420),
            "affine-to-perspective": lambda: (media.affine_to_perspective(), 420),
            "forward-vs-backward": lambda: (media.forward_vs_backward(s), 520),
        }
        for name in only:
            (frames, fps), width = table[name]()
            media.encode(frames, fps, media.MEDIA_DIR / f"{name}.webp", width)
    else:
        media.render_all()


if __name__ == "__main__":
    align_media()
