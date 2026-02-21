# Energy Meter OCR Pipeline

Automatically photograph a **Landis+Gyr** (or similar 7-segment LCD) energy
meter with a Raspberry Pi Camera Module, extract the numeric reading using
OpenCV + Tesseract, and store every hourly reading in a local SQLite database.

---

## Features

| | |
|---|---|
| 📷 | Pi Camera Module 2/3 via **picamera2** |
| 🔍 | Automatic **perspective correction** – handles camera tilt |
| 🔢 | **Multi-pass Tesseract OCR** with per-digit voting (calibrated for 7-segment LCD) |
| 🗄️ | **SQLite** database – no server needed |
| ✅ | **Sanity checking** – flags impossible readings (decrease, huge jump) |
| ⏰ | **Hourly systemd timer** – runs on boot, persists through reboots |
| 🗑️ | Images deleted after processing (configurable) |
| 🐛 | Debug mode saves every preprocessing step as images |
| 🎯 | `--calibrate` mode to tune the display crop region |

---

## Hardware requirements

* Raspberry Pi (any model with a CSI camera connector)
* Raspberry Pi Camera Module 2 or 3 (NoIR also works)
* The camera should be mounted **roughly level** with the meter display,
  approximately 20–40 cm away.  Perspective tilt up to ≈15° is corrected
  automatically; more than that requires setting `display.crop_region`.

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/your-repo/energy-monitor
cd energy-monitor
sudo bash scripts/install.sh
```

`install.sh` installs system packages, creates a Python virtualenv, and
registers the systemd timer.

### 2. Test with the sample image (no Pi needed)

```bash
venv/bin/python -m src.pipeline --image energy_reading.png
# → 35.768 MWh
```

### 3. Calibrate (recommended before first production run)

Point the camera at the meter, take a photo (or use an existing one), then:

```bash
venv/bin/python -m src.pipeline --image energy_reading.png --calibrate
```

This prints the detected display coordinates and saves `debug/calibration.png`
with a green rectangle around the display.

If the auto-detection finds the wrong region, you can hard-code the crop:

```yaml
# config.yaml
display:
  crop_region: [480, 408, 305, 75]   # [x, y, width, height]
```

### 4. Check stored readings

```bash
venv/bin/python -m src.pipeline --list
venv/bin/python -m src.pipeline --export readings.csv
```

---

## Project structure

```
energy-monitor/
├── src/
│   ├── capture.py      # picamera2 capture (or file load for testing)
│   ├── preprocess.py   # perspective detection + grayscale extraction
│   ├── ocr.py          # multi-pass Tesseract with per-digit voting
│   ├── database.py     # SQLite CRUD
│   └── pipeline.py     # orchestrator + CLI
├── scripts/
│   ├── install.sh              # one-shot Pi setup script
│   ├── energy-monitor.service  # systemd service (oneshot)
│   └── energy-monitor.timer    # systemd timer (hourly)
├── config.yaml         # all tunable parameters
├── requirements.txt
└── README.md
```

---

## Configuration reference

```yaml
# config.yaml (key settings)

camera:
  resolution: [1920, 1080]
  rotation: 0              # 0 / 90 / 180 / 270

display:
  crop_region: null        # set to [x, y, w, h] after --calibrate
  roi_y_start: 0.35        # ROI fractions for auto-detection
  roi_y_end:   0.70
  roi_x_start: 0.25
  roi_x_end:   0.75
  digit_crop_y: [0.18, 0.92]   # digit area within the warped display
  digit_crop_x: [0.14, 0.85]

ocr:
  decimal_places: 3        # reading = 0035768 → 35.768
  total_digits: 7          # full display width in digits

database:
  path: "data/readings.db"
  unit: "MWh"

pipeline:
  image_dir: "/tmp/energy_monitor"
  delete_images: true      # set false to keep images
  save_debug_images: false
  log_file: "data/energy_monitor.log"
```

---

## OCR approach

The pipeline uses **five Tesseract passes** per image, each with different
CLAHE clip / scale / threshold parameters.  Results are combined by
**per-digit voting**, with tiebreakers for common 7-segment confusion pairs:

| Winner | Beats | Reason |
|--------|-------|--------|
| `7` | `1` | Top horizontal bar is the faintest segment |
| `0` | `8` | Middle bar absent in `0` appears as noise |
| `0` | `9` | Same middle-bar noise pattern |
| `3` | `9` | `9` = `3` + faint top-left bar |

---

## Systemd management

```bash
# Timer status
sudo systemctl status energy-monitor.timer

# Trigger a reading immediately
sudo systemctl start energy-monitor.service

# View logs
sudo journalctl -u energy-monitor.service -f

# Disable automatic runs
sudo systemctl disable energy-monitor.timer
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| "Display not detected" | Run `--calibrate` and set `display.crop_region` |
| Reading flagged insane | Check `--list` for trend; adjust `ocr.max_jump` |
| Low confidence (< 60) | Improve lighting; ensure camera is level and in focus |
| Always wrong digit | Run `--debug` and inspect images in `debug/` |
| Camera not found | Check `sudo raspi-config → Interface Options → Camera` |
