"""
ascii_camera.py - ASCII art camera with multiple render modes.

Usage:
    python ascii_camera.py [options]

Options:
    -c, --camera     Camera index (default: 0)
    -w, --width      Deprecated; ASCII width is fixed at 220
    -m, --mode       Render mode: density | edge | color (default: density)
    --no-window      Print ASCII to terminal instead of OpenCV window
    --font-scale     Font scale for OpenCV window (default: 0.35)

Keyboard shortcuts (OpenCV window):
    ESC       Quit
    D         Switch to density mode
    E         Switch to edge mode
    C         Switch to color mode
    T         Toggle terminal output
    +/-       Increase/decrease brightness
    [ / ]     Decrease/increase contrast
    B         Reset brightness
    X         Reset contrast
    S         Save current frame as PNG
"""

import argparse
import os
import sys
import time
import cv2
import numpy as np

WINDOW_NAME = "ASCII Camera"
FIXED_ASCII_WIDTH = 220
DEFAULT_BRIGHTNESS = 0.0
DEFAULT_CONTRAST = 1.0
MAX_CAMERA_SCAN = 8

# ---------------------------------------------------------------------------
# ASCII character sets
# ---------------------------------------------------------------------------

# Density ramp for black-background display: dark pixels stay quiet, bright
# pixels become stronger visible characters.
ASCII_DENSITY = (
    " .'`^\",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
)

# Lighter set for edge mode; thin strokes look better with sparse chars.
ASCII_EDGES = "  .:-=+*#%@"

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Measured character cell size for FONT_HERSHEY_SIMPLEX at scale 0.35, thickness 1.
# Adjust if text appears misaligned on your system.
CHAR_W = 7   # pixels per character (width)
CHAR_H = 10  # pixels per character (height)

# Camera UI palette, in OpenCV's BGR color order.
UI_BG = (24, 12, 4)
UI_PANEL = (42, 22, 8)
UI_PANEL_ALT = (66, 34, 12)
UI_BORDER = (150, 95, 34)
UI_GREEN = (220, 145, 60)
UI_GREEN_DIM = (128, 82, 34)
UI_GREEN_DARK = (72, 45, 18)
UI_TEXT = (216, 220, 224)
UI_WARNING = (210, 185, 90)
ASCII_MONO_COLOR = (200, 200, 200)

# Aspect-ratio correction: monospace cells are ~CHAR_H/CHAR_W taller than wide.
ASPECT_CORRECTION = CHAR_W / CHAR_H  # approx 0.7


# ---------------------------------------------------------------------------
# Frame processing helpers
# ---------------------------------------------------------------------------

def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """Apply CLAHE for adaptive local contrast enhancement."""
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(gray)


def prepare_gray_for_ascii(gray: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
    """Clean and normalize a grayscale frame so ASCII details read clearly."""
    adjusted = np.clip(gray.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)
    denoised = cv2.bilateralFilter(adjusted, 5, 45, 45)
    enhanced = enhance_contrast(denoised)
    sharpened = cv2.addWeighted(enhanced, 1.35, cv2.GaussianBlur(enhanced, (0, 0), 1.1), -0.35, 0)
    return cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)


def resize_for_ascii(frame: np.ndarray, width: int) -> np.ndarray:
    """Resize a grayscale frame to `width` columns, correcting for cell aspect ratio."""
    h, w = frame.shape
    height = int((h / w) * width * ASPECT_CORRECTION)
    return cv2.resize(frame, (width, max(1, height)), interpolation=cv2.INTER_AREA)


def frame_to_density_ascii(gray: np.ndarray, brightness: float, contrast: float) -> list[str]:
    """Map pixel intensities to ASCII density characters."""
    prepared = prepare_gray_for_ascii(gray, brightness, contrast)
    small = resize_for_ascii(prepared, _state["width"])
    small = cv2.convertScaleAbs(small, alpha=1.12, beta=4)

    n = len(ASCII_DENSITY) - 1
    indexes = (small.astype(np.uint32) * n // 255).clip(0, n)
    return ["".join(ASCII_DENSITY[i] for i in row) for row in indexes]


def frame_to_edge_ascii(gray: np.ndarray, brightness: float, contrast: float) -> list[str]:
    """Use Canny edge detection and map edge strength to ASCII."""
    prepared = prepare_gray_for_ascii(gray, brightness, contrast)
    blurred = cv2.GaussianBlur(prepared, (3, 3), 0)
    edges = cv2.Canny(blurred, threshold1=40, threshold2=120)
    small = resize_for_ascii(edges, _state["width"])

    n = len(ASCII_EDGES) - 1
    indexes = (small.astype(np.uint32) * n // 255).clip(0, n)
    return ["".join(ASCII_EDGES[i] for i in row) for row in indexes]


def frame_to_color_ascii(
    bgr: np.ndarray, brightness: float, contrast: float
) -> tuple[list[str], np.ndarray]:
    """
    Density ASCII mapping, but also return a downsampled color array so each
    character can be rendered in its source pixel's color.

    Returns:
        ascii_lines  - list of strings (same as density mode)
        colors       - (rows, cols, 3) uint8 BGR array
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    prepared = prepare_gray_for_ascii(gray, brightness, contrast)
    small_gray = resize_for_ascii(prepared, _state["width"])
    small_gray = cv2.convertScaleAbs(small_gray, alpha=1.12, beta=4)

    h, w = small_gray.shape
    small_color = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    # Boost saturation for visual impact
    hsv = cv2.cvtColor(small_color, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.4, 0, 255)
    small_color = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    n = len(ASCII_DENSITY) - 1
    indexes = (small_gray.astype(np.uint32) * n // 255).clip(0, n)
    lines = ["".join(ASCII_DENSITY[i] for i in row) for row in indexes]
    return lines, small_color


def open_camera(index: int) -> cv2.VideoCapture:
    """Open a camera and request the preferred capture size."""
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def find_available_cameras(max_index: int = MAX_CAMERA_SCAN) -> list[int]:
    """Probe common camera indexes and return the ones that can produce a frame."""
    cameras: list[int] = []
    for index in range(max_index):
        cap = open_camera(index)
        if cap.isOpened():
            ok, _frame = cap.read()
            if ok:
                cameras.append(index)
        cap.release()
    return cameras


def get_pictures_dir() -> str:
    """Return the user's Pictures folder, falling back to the home directory."""
    home = os.path.expanduser("~")
    pictures = os.path.join(home, "Pictures")
    return pictures if os.path.isdir(pictures) else home


def build_capture_path(save_dir: str, mode: str) -> str:
    """Create a capture path using the current filter name."""
    filter_name = mode.upper()
    base_name = f"{filter_name} asciicam"
    candidate = os.path.join(save_dir, f"{base_name}.png")
    if not os.path.exists(candidate):
        return candidate

    counter = 2
    while True:
        candidate = os.path.join(save_dir, f"{base_name} {counter}.png")
        if not os.path.exists(candidate):
            return candidate
        counter += 1


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def draw_text(
    image: np.ndarray,
    text: str,
    pos: tuple[int, int],
    scale: float = 0.45,
    color: tuple[int, int, int] = UI_TEXT,
    thickness: int = 1,
) -> None:
    """Small wrapper so UI text stays visually consistent."""
    cv2.putText(image, text, pos, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_panel(
    image: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    label: str | None = None,
    fill: tuple[int, int, int] = UI_PANEL,
) -> None:
    """Draw a crisp camera-control panel with a subtle navy border."""
    cv2.rectangle(image, (x, y), (x + w, y + h), fill, -1)
    cv2.rectangle(image, (x, y), (x + w, y + h), UI_GREEN_DARK, 1)
    cv2.rectangle(image, (x + 2, y + 2), (x + w - 2, y + h - 2), UI_BORDER, 1)
    if label:
        draw_text(image, label, (x + 12, y + 21), 0.43, UI_GREEN, 1)


def draw_button(
    image: np.ndarray,
    rect: tuple[int, int, int, int],
    label: str,
    active: bool = False,
) -> None:
    """Draw a compact camera-app style button."""
    x, y, w, h = rect
    fill = UI_PANEL_ALT if active else UI_PANEL
    border = UI_GREEN if active else UI_GREEN_DARK
    cv2.rectangle(image, (x, y), (x + w, y + h), fill, -1)
    cv2.rectangle(image, (x, y), (x + w, y + h), border, 1)
    text_size, _ = cv2.getTextSize(label, FONT, 0.38, 1)
    tx = x + max(8, (w - text_size[0]) // 2)
    ty = y + (h + text_size[1]) // 2 - 2
    draw_text(image, label, (tx, ty), 0.38, UI_GREEN if active else UI_TEXT, 1)


def draw_meter(
    image: np.ndarray,
    x: int,
    y: int,
    w: int,
    label: str,
    value_text: str,
    ratio: float,
) -> None:
    """Draw a horizontal control meter."""
    ratio = float(np.clip(ratio, 0.0, 1.0))
    draw_text(image, label, (x, y), 0.36, UI_GREEN_DIM, 1)
    draw_text(image, value_text, (x + w - 56, y), 0.36, UI_TEXT, 1)
    track_y = y + 11
    cv2.rectangle(image, (x, track_y), (x + w, track_y + 7), UI_GREEN_DARK, -1)
    cv2.rectangle(image, (x, track_y), (x + int(w * ratio), track_y + 7), UI_GREEN, -1)


def draw_slider(
    image: np.ndarray,
    x: int,
    y: int,
    w: int,
    label: str,
    value_text: str,
    ratio: float,
    active: bool = False,
) -> tuple[int, int, int, int]:
    """Draw a draggable slider and return its hit rectangle."""
    ratio = float(np.clip(ratio, 0.0, 1.0))
    draw_text(image, label, (x, y), 0.36, UI_GREEN_DIM, 1)
    draw_text(image, value_text, (x + w - 56, y), 0.36, UI_TEXT, 1)

    track_y = y + 18
    handle_x = x + int(w * ratio)
    track_color = UI_GREEN if active else UI_GREEN_DARK
    cv2.rectangle(image, (x, track_y), (x + w, track_y + 8), UI_GREEN_DARK, -1)
    cv2.rectangle(image, (x, track_y), (handle_x, track_y + 8), UI_GREEN, -1)
    cv2.circle(image, (handle_x, track_y + 4), 9, track_color, -1)
    cv2.circle(image, (handle_x, track_y + 4), 10, UI_TEXT if active else UI_GREEN_DIM, 1)
    return (x - 8, track_y - 12, w + 16, 32)


def set_slider_value(name: str, x: int) -> None:
    """Update one slider value from a mouse x coordinate."""
    slider = _state["ui_sliders"].get(name)
    if not slider:
        return

    track_x, _track_y, track_w, _track_h = slider["track"]
    ratio = float(np.clip((x - track_x) / track_w, 0.0, 1.0))
    if name == "brightness":
        _state["brightness"] = round(-100 + ratio * 200)
    elif name == "contrast":
        _state["contrast"] = round(0.1 + ratio * 2.9, 1)


def point_in_rect(x: int, y: int, rect: tuple[int, int, int, int]) -> bool:
    """Return whether a point is inside an x, y, width, height rectangle."""
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh


def handle_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
    """Handle clickable camera controls inside the OpenCV window."""
    if event == cv2.EVENT_LBUTTONDOWN:
        for action, rect in _state["ui_action_buttons"].items():
            if point_in_rect(x, y, rect):
                if action == "reset_brightness":
                    _state["brightness"] = DEFAULT_BRIGHTNESS
                elif action == "reset_contrast":
                    _state["contrast"] = DEFAULT_CONTRAST
                elif action == "settings":
                    _state["show_settings"] = not _state["show_settings"]
                elif action == "quit":
                    _state["quit_requested"] = True
                return

        for camera_index, rect in _state["ui_camera_buttons"].items():
            if point_in_rect(x, y, rect):
                _state["requested_camera"] = camera_index
                _state["show_settings"] = False
                return

        for mode, rect in _state["ui_buttons"].items():
            if point_in_rect(x, y, rect):
                _state["mode"] = mode
                return

        if _state["ui_shutter"] and point_in_rect(x, y, _state["ui_shutter"]):
            _state["save_requested"] = True
            return

        for name, slider in _state["ui_sliders"].items():
            if point_in_rect(x, y, slider["hit"]):
                _state["active_slider"] = name
                set_slider_value(name, x)
                return

    elif event == cv2.EVENT_MOUSEMOVE and _state["active_slider"]:
        set_slider_value(_state["active_slider"], x)

    elif event == cv2.EVENT_LBUTTONUP:
        if _state["active_slider"]:
            set_slider_value(_state["active_slider"], x)
        _state["active_slider"] = None


def enforce_window_bounds(image: np.ndarray) -> None:
    """Keep the window from shrinking or distorting the camera UI."""
    native_h, native_w = image.shape[:2]
    try:
        _x, _y, window_w, window_h = cv2.getWindowImageRect(WINDOW_NAME)
    except cv2.error:
        return

    if window_w <= 0 or window_h <= 0:
        return

    scale = max(window_w / native_w, window_h / native_h, 1.0)
    target_w = int(round(native_w * scale))
    target_h = int(round(native_h * scale))

    if abs(target_w - window_w) > 2 or abs(target_h - window_h) > 2:
        cv2.resizeWindow(WINDOW_NAME, target_w, target_h)


def draw_viewfinder_guides(image: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    """Draw camera framing corner marks."""
    corner = 46
    line = UI_GREEN
    cv2.line(image, (x, y), (x + corner, y), line, 2)
    cv2.line(image, (x, y), (x, y + corner), line, 2)
    cv2.line(image, (x + w, y), (x + w - corner, y), line, 2)
    cv2.line(image, (x + w, y), (x + w, y + corner), line, 2)
    cv2.line(image, (x, y + h), (x + corner, y + h), line, 2)
    cv2.line(image, (x, y + h), (x, y + h - corner), line, 2)
    cv2.line(image, (x + w, y + h), (x + w - corner, y + h), line, 2)
    cv2.line(image, (x + w, y + h), (x + w, y + h - corner), line, 2)


def draw_matrix_rain(image: np.ndarray, tick: int) -> None:
    """Add low-contrast falling code to the unused UI background."""
    chars = "0101010011010110"
    h, w = image.shape[:2]
    for x in range(20, w, 34):
        offset = (tick * 3 + x * 7) % max(1, h)
        for i in range(0, h, 24):
            y = (i + offset) % h
            ch = chars[(x + i + tick) % len(chars)]
            draw_text(image, ch, (x, y), 0.35, (38, 22, 8), 1)


def render_ascii_image(
    ascii_lines: list[str],
    font_scale: float,
    colors: np.ndarray | None = None,
    fps: float = 0.0,
    mode: str = "density",
    brightness: float = 0,
    contrast: float = 1.0,
) -> np.ndarray:
    """Draw ASCII camera output inside a camera app UI."""
    rows = len(ascii_lines)
    cols = len(ascii_lines[0]) if rows else 1

    feed_pad = 12
    top_h = 78
    bottom_h = 94
    side_w = 220
    outer_pad = 18
    capacity_cols = FIXED_ASCII_WIDTH
    fallback_rows = int(np.ceil((rows / max(cols, 1)) * FIXED_ASCII_WIDTH))
    capacity_rows = max(rows, _state.get("view_rows_capacity") or fallback_rows)
    feed_w = capacity_cols * CHAR_W + feed_pad * 2
    feed_h = capacity_rows * CHAR_H + feed_pad * 2
    img_w = feed_w + side_w + outer_pad * 3
    img_h = feed_h + top_h + bottom_h + outer_pad * 2

    image = np.full((img_h, img_w, 3), UI_BG, dtype=np.uint8)
    draw_matrix_rain(image, _state["ui_tick"])
    _state["ui_buttons"] = {}
    _state["ui_sliders"] = {}
    _state["ui_action_buttons"] = {}
    _state["ui_camera_buttons"] = {}
    _state["ui_shutter"] = None

    feed_x = outer_pad
    feed_y = top_h + outer_pad
    side_x = feed_x + feed_w + outer_pad

    draw_panel(image, feed_x - 6, feed_y - 6, feed_w + 12, feed_h + 12, fill=(1, 12, 5))
    cv2.rectangle(image, (feed_x, feed_y), (feed_x + feed_w, feed_y + feed_h), (0, 0, 0), -1)

    ascii_x = feed_x + feed_pad + ((capacity_cols - cols) * CHAR_W) // 2
    ascii_y = feed_y + feed_pad + ((capacity_rows - rows) * CHAR_H) // 2
    for r, line in enumerate(ascii_lines):
        y = ascii_y + (r + 1) * CHAR_H
        for c, ch in enumerate(line):
            if ch == " ":
                continue
            x = ascii_x + c * CHAR_W
            color = (
                (int(colors[r, c, 0]), int(colors[r, c, 1]), int(colors[r, c, 2]))
                if colors is not None
                else ASCII_MONO_COLOR
            )
            cv2.putText(image, ch, (x, y), FONT, font_scale, color, 1, cv2.LINE_AA)

    draw_viewfinder_guides(image, feed_x, feed_y, feed_w, feed_h)

    draw_panel(image, outer_pad, outer_pad, img_w - outer_pad * 2, 56, fill=UI_PANEL)
    settings_rect = (outer_pad + 18, outer_pad + 12, 112, 32)
    close_rect = (img_w - outer_pad - 46, outer_pad + 12, 32, 32)
    draw_button(image, settings_rect, "SETTINGS", _state["show_settings"])
    draw_button(image, close_rect, "X", False)
    _state["ui_action_buttons"]["settings"] = settings_rect
    _state["ui_action_buttons"]["quit"] = close_rect
    draw_text(image, f"FPS {fps:04.1f}", (img_w - 342, outer_pad + 35), 0.48, UI_TEXT, 1)
    draw_text(image, time.strftime("%H:%M:%S"), (img_w - 190, outer_pad + 35), 0.52, UI_TEXT, 1)

    draw_panel(image, side_x, feed_y - 6, side_w, feed_h + 12, "CONTROLS", fill=UI_PANEL)
    button_y = feed_y + 36
    for label, key in (("DENSITY", "density"), ("EDGE", "edge"), ("COLOR", "color")):
        rect = (side_x + 18, button_y, side_w - 36, 34)
        draw_button(image, rect, label, mode == key)
        _state["ui_buttons"][key] = rect
        button_y += 44

    slider_x = side_x + 20
    slider_w = side_w - 40
    slider_y = button_y + 18
    slider_defs = (
        ("brightness", "BRIGHTNESS", f"{brightness:+.0f}", (brightness + 100) / 200),
        ("contrast", "CONTRAST", f"{contrast:.1f}x", (contrast - 0.1) / 2.9),
    )
    for name, label, value_text, ratio in slider_defs:
        hit = draw_slider(
            image,
            slider_x,
            slider_y,
            slider_w,
            label,
            value_text,
            ratio,
            _state["active_slider"] == name,
        )
        _state["ui_sliders"][name] = {
            "hit": hit,
            "track": (slider_x, slider_y + 18, slider_w, 8),
        }
        slider_y += 58

    reset_y = slider_y + 4
    reset_b_rect = (side_x + 18, reset_y, side_w - 36, 32)
    reset_c_rect = (side_x + 18, reset_y + 42, side_w - 36, 32)
    draw_button(image, reset_b_rect, "RESET BRIGHTNESS", False)
    draw_button(image, reset_c_rect, "RESET CONTRAST", False)
    _state["ui_action_buttons"]["reset_brightness"] = reset_b_rect
    _state["ui_action_buttons"]["reset_contrast"] = reset_c_rect

    bottom_y = feed_y + feed_h + outer_pad
    draw_panel(image, outer_pad, bottom_y, img_w - outer_pad * 2, bottom_h - outer_pad, fill=UI_PANEL)
    shutter_cx = img_w // 2
    shutter_cy = bottom_y + 38
    cv2.circle(image, (shutter_cx, shutter_cy), 31, UI_GREEN_DARK, 2)
    cv2.circle(image, (shutter_cx, shutter_cy), 22, UI_GREEN, 2)
    draw_text(image, "S", (shutter_cx - 7, shutter_cy + 7), 0.6, UI_TEXT, 2)
    draw_text(image, "SAVE FRAME", (shutter_cx - 54, shutter_cy + 52), 0.38, UI_GREEN_DIM, 1)
    _state["ui_shutter"] = (shutter_cx - 34, shutter_cy - 34, 68, 68)

    save_msg = _state.get("last_saved", "")
    save_age = time.time() - _state.get("last_saved_at", 0.0)
    if save_msg and save_age < 3.0:
        draw_text(image, save_msg, (shutter_cx + 92, bottom_y + 45), 0.4, UI_WARNING, 1)
    else:
        draw_text(image, "READY", (shutter_cx + 92, bottom_y + 45), 0.45, UI_GREEN, 1)

    if _state["show_settings"]:
        panel_w = 520
        available_cameras = _state["available_cameras"] or [_state["camera_index"]]
        panel_h = 104 + len(available_cameras) * 52
        panel_x = (img_w - panel_w) // 2
        panel_y = (img_h - panel_h) // 2
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (img_w, img_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.42, image, 0.58, 0, image)
        draw_panel(image, panel_x, panel_y, panel_w, panel_h, "CAMERA SETTINGS", fill=UI_PANEL)
        draw_text(image, f"CURRENT CAMERA: {_state['camera_index']}", (panel_x + 28, panel_y + 58), 0.5, UI_TEXT, 1)
        button_y = panel_y + 88
        for camera_index in available_cameras:
            rect = (panel_x + 28, button_y, panel_w - 56, 38)
            draw_button(image, rect, f"CAMERA {camera_index}", camera_index == _state["camera_index"])
            _state["ui_camera_buttons"][camera_index] = rect
            button_y += 52

    _state["ui_tick"] = (_state["ui_tick"] + 1) % 10_000
    return image


def print_ascii_terminal(ascii_lines: list[str]) -> None:
    """Clear terminal and print ASCII art."""
    sys.stdout.write("\033[2J\033[H")  # clear + home
    sys.stdout.write("\n".join(ascii_lines) + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Mutable program state (kept in a dict to avoid globals scattered everywhere)
# ---------------------------------------------------------------------------

_state: dict = {
    "mode": "density",
    "width": FIXED_ASCII_WIDTH,
    "brightness": DEFAULT_BRIGHTNESS,
    "contrast": DEFAULT_CONTRAST,
    "terminal": False,
    "save_requested": False,
    "font_scale": 0.35,
    "ui_tick": 0,
    "last_saved": "",
    "last_saved_at": 0.0,
    "active_slider": None,
    "ui_buttons": {},
    "ui_sliders": {},
    "ui_action_buttons": {},
    "ui_camera_buttons": {},
    "ui_shutter": None,
    "view_rows_capacity": None,
    "show_settings": False,
    "available_cameras": [],
    "camera_index": 0,
    "requested_camera": None,
    "quit_requested": False,
}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ASCII Camera - real-time ASCII art from your webcam.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-c", "--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument(
        "-w",
        "--width",
        type=int,
        default=FIXED_ASCII_WIDTH,
        help="Deprecated; ASCII width is fixed at 220",
    )
    parser.add_argument(
        "-m", "--mode",
        choices=["density", "edge", "color"],
        default="density",
        help="Render mode (default: density)",
    )
    parser.add_argument("--no-window", action="store_true", help="Output to terminal instead of window")
    parser.add_argument("--font-scale", type=float, default=0.35, help="OpenCV font scale (default: 0.35)")
    args = parser.parse_args()

    _state["mode"] = args.mode
    _state["width"] = FIXED_ASCII_WIDTH
    _state["terminal"] = args.no_window
    _state["font_scale"] = args.font_scale

    print("Scanning cameras...")
    available_cameras = find_available_cameras()
    if args.camera not in available_cameras:
        available_cameras.append(args.camera)
        available_cameras = sorted(set(available_cameras))
    _state["available_cameras"] = available_cameras
    _state["camera_index"] = args.camera

    cap = open_camera(args.camera)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {args.camera}.")
        print("Try a different index with -c 1")
        sys.exit(1)

    if not _state["terminal"]:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setMouseCallback(WINDOW_NAME, handle_mouse)

    print("ASCII Camera running. Press Q or ESC to quit. Click controls or drag sliders in the window.")

    prev_time = time.perf_counter()
    fps = 0.0
    frame_count = 0
    save_dir = os.path.join(get_pictures_dir(), "PYCAM")
    os.makedirs(save_dir, exist_ok=True)

    while True:
        requested_camera = _state.get("requested_camera")
        if requested_camera is not None and requested_camera != _state["camera_index"]:
            new_cap = open_camera(int(requested_camera))
            if new_cap.isOpened():
                cap.release()
                cap = new_cap
                _state["camera_index"] = int(requested_camera)
                _state["last_saved"] = f"CAMERA {_state['camera_index']} SELECTED"
                _state["last_saved_at"] = time.time()
            else:
                new_cap.release()
                _state["last_saved"] = f"CAMERA {requested_camera} UNAVAILABLE"
                _state["last_saved_at"] = time.time()
            _state["requested_camera"] = None

        ok, frame = cap.read()
        if not ok:
            print("Error: Failed to read frame.")
            break

        frame_h, frame_w = frame.shape[:2]
        if frame_w:
            _state["view_rows_capacity"] = int(np.ceil((frame_h / frame_w) * FIXED_ASCII_WIDTH * ASPECT_CORRECTION))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mode = _state["mode"]
        brightness = _state["brightness"]
        contrast = _state["contrast"]

        colors = None
        if mode == "density":
            ascii_lines = frame_to_density_ascii(gray, brightness, contrast)
        elif mode == "edge":
            ascii_lines = frame_to_edge_ascii(gray, brightness, contrast)
        else:  # color
            ascii_lines, colors = frame_to_color_ascii(frame, brightness, contrast)

        # FPS
        now = time.perf_counter()
        frame_count += 1
        if frame_count % 10 == 0:
            fps = 10 / (now - prev_time)
            prev_time = now

        if _state["terminal"]:
            print_ascii_terminal(ascii_lines)
        else:
            img = render_ascii_image(
                ascii_lines, _state["font_scale"], colors, fps, mode, brightness, contrast
            )

            if _state["save_requested"]:
                fname = build_capture_path(save_dir, mode)
                cv2.imwrite(fname, img)
                print(f"Saved: {fname}")
                _state["last_saved"] = f"SAVED {os.path.basename(fname)}"
                _state["last_saved_at"] = time.time()
                _state["save_requested"] = False

            cv2.imshow(WINDOW_NAME, img)
            enforce_window_bounds(img)

            key = cv2.waitKey(1) & 0xFF
            try:
                window_visible = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1
            except cv2.error:
                window_visible = False

            if key == 27 or not window_visible or _state["quit_requested"]:  # ESC, window close, or top X
                break
            elif key == ord("d"):
                _state["mode"] = "density"
            elif key == ord("e"):
                _state["mode"] = "edge"
            elif key == ord("c"):
                _state["mode"] = "color"
            elif key == ord("t"):
                _state["terminal"] = not _state["terminal"]
            elif key == ord("+") or key == ord("="):
                _state["brightness"] = min(_state["brightness"] + 10, 100)
            elif key == ord("-"):
                _state["brightness"] = max(_state["brightness"] - 10, -100)
            elif key == ord("]"):
                _state["contrast"] = min(round(_state["contrast"] + 0.1, 1), 3.0)
            elif key == ord("["):
                _state["contrast"] = max(round(_state["contrast"] - 0.1, 1), 0.1)
            elif key == ord("b"):
                _state["brightness"] = DEFAULT_BRIGHTNESS
            elif key == ord("x"):
                _state["contrast"] = DEFAULT_CONTRAST
            elif key == ord("s"):
                _state["save_requested"] = True

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
