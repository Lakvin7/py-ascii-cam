# py-ascii-cam

Real-time Python webcam app that turns your camera feed into ASCII art. It includes an OpenCV window with clickable controls, multiple render modes, brightness and contrast sliders, camera switching, terminal output, and one-click PNG capture.

## Features

- Live webcam-to-ASCII rendering
- Density, edge, and color ASCII modes
- Clickable OpenCV camera UI with mode buttons, sliders, settings, and shutter
- Keyboard shortcuts for fast control
- Camera scanning and switching
- Optional terminal-only ASCII output
- PNG capture saved to your Pictures folder

## Requirements

- Python 3.10 or newer recommended
- A working webcam
- Windows, macOS, or Linux with OpenCV camera support

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run

Start the app with the default camera:

```bash
python ascii_camera.py
```

On Windows, you can also run:

```bat
run_windows.bat
```

Use a different camera index:

```bash
python ascii_camera.py --camera 1
```

Start in a specific render mode:

```bash
python ascii_camera.py --mode edge
python ascii_camera.py --mode color
```

Print ASCII directly in the terminal instead of opening the OpenCV window:

```bash
python ascii_camera.py --no-window
```

## Controls

### Window Controls

- `DENSITY`, `EDGE`, `COLOR`: switch render mode
- `BRIGHTNESS` slider: adjust image brightness
- `CONTRAST` slider: adjust contrast
- `RESET BRIGHTNESS`: restore default brightness
- `RESET CONTRAST`: restore default contrast
- `SETTINGS`: choose from detected cameras
- Shutter button: save the current ASCII frame as a PNG
- `X`: quit

### Keyboard Shortcuts

| Key | Action |
| --- | --- |
| `Esc` | Quit |
| `D` | Density mode |
| `E` | Edge mode |
| `C` | Color mode |
| `T` | Toggle terminal output |
| `+` / `-` | Increase/decrease brightness |
| `]` / `[` | Increase/decrease contrast |
| `B` | Reset brightness |
| `X` | Reset contrast |
| `S` | Save current frame |

## Captures

Saved frames are written to:

```text
~/Pictures/PYCAM
```

If the Pictures folder is not available, the app falls back to your home directory. Filenames include the active mode, such as `DENSITY asciicam.png` or `COLOR asciicam 2.png`.

## Command-Line Options

```text
-c, --camera       Camera index (default: 0)
-w, --width        Deprecated; ASCII width is fixed at 220
-m, --mode         Render mode: density, edge, or color
--no-window        Print ASCII to terminal instead of the OpenCV window
--font-scale       Font scale for the OpenCV window text
```

## Troubleshooting

If the app cannot open your camera, try another index:

```bash
python ascii_camera.py --camera 1
python ascii_camera.py --camera 2
```

Close other apps that may be using the webcam, then restart `py-ascii-cam`.
