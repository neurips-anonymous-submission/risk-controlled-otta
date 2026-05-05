from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/timesbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def fit_image(image: Image.Image, box_w: int, box_h: int) -> Image.Image:
    image = image.convert("RGB")
    scale = min(box_w / image.width, box_h / image.height)
    new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def paste_center(canvas: Image.Image, image: Image.Image, x: int, y: int, w: int, h: int) -> None:
    fitted = fit_image(image, w, h)
    off_x = x + (w - fitted.width) // 2
    off_y = y + (h - fitted.height) // 2
    canvas.paste(fitted, (off_x, off_y))


def draw_column(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    x: int,
    top_y: int,
    panel_w: int,
    top_h: int,
    gap_y: int,
    bottom_h: int,
    title: str,
    panel_tag: str,
    full_image: Path,
    zoom_image: Path,
    title_font,
    tag_font,
    border_color=(220, 220, 220),
) -> None:
    pad = 6
    draw.text((x + panel_w // 2, top_y - 38), title, fill="black", font=title_font, anchor="mm")

    top_box = (x, top_y, x + panel_w, top_y + top_h)
    bottom_y = top_y + top_h + gap_y
    bottom_box = (x, bottom_y, x + panel_w, bottom_y + bottom_h)

    draw.rectangle(top_box, outline=border_color, width=2)
    draw.rectangle(bottom_box, outline=border_color, width=2)

    paste_center(canvas, Image.open(full_image), x + pad, top_y + pad, panel_w - 2 * pad, top_h - 2 * pad)
    paste_center(canvas, Image.open(zoom_image), x + pad, bottom_y + pad, panel_w - 2 * pad, bottom_h - 2 * pad)

    draw.text((x + panel_w // 2, bottom_y + bottom_h + 30), panel_tag, fill="black", font=tag_font, anchor="mm")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_full", type=str, required=True)
    parser.add_argument("--otta_full", type=str, required=True)
    parser.add_argument("--ours_full", type=str, required=True)
    parser.add_argument("--source_zoom", type=str, required=True)
    parser.add_argument("--otta_zoom", type=str, required=True)
    parser.add_argument("--ours_zoom", type=str, required=True)
    parser.add_argument(
        "--output_path",
        type=str,
        default="visualization_results_dinov3_heatmap/qualitative_comparison_main.png",
    )
    args = parser.parse_args()

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    canvas_w = 1860
    canvas_h = 980
    margin_x = 58
    margin_top = 70
    margin_bottom = 90
    col_gap = 34
    row_gap = 18
    panel_w = (canvas_w - 2 * margin_x - 2 * col_gap) // 3
    panel_h_total = canvas_h - margin_top - margin_bottom
    top_h = 360
    bottom_h = panel_h_total - top_h - row_gap

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(34)
    tag_font = load_font(30)

    cols = [
        ("Source-only / No TTA", "(a)", Path(args.source_full), Path(args.source_zoom)),
        ("Continuous OTTA", "(b)", Path(args.otta_full), Path(args.otta_zoom)),
        ("Risk-Controlled-OTTA (Dual-Branch)", "(c)", Path(args.ours_full), Path(args.ours_zoom)),
    ]

    for idx, (title, tag, full_path, zoom_path) in enumerate(cols):
        x = margin_x + idx * (panel_w + col_gap)
        draw_column(
            canvas=canvas,
            draw=draw,
            x=x,
            top_y=margin_top,
            panel_w=panel_w,
            top_h=top_h,
            gap_y=row_gap,
            bottom_h=bottom_h,
            title=title,
            panel_tag=tag,
            full_image=full_path,
            zoom_image=zoom_path,
            title_font=title_font,
            tag_font=tag_font,
        )

    canvas.save(out_path, quality=95)
    print(f"Saved qualitative figure to: {out_path}")


if __name__ == "__main__":
    main()


