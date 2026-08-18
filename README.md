# I2DMS (Intelligent In-Cabin Driver Monitoring System)

I2DMS is a real-time computer vision system that monitors driver distraction (phone usage) using a hybrid **YOLOv8 + MediaPipe** architecture, integrated with vehicle speed threshold logic and a **NiceGUI** telemetry dashboard.

---
## Key Features

* **Hybrid Detection:** Combines Ultralytics YOLOv8 object detection with MediaPipe Hands fallback tracking to handle occlusion (e.g., when a hand covers a phone held against the ear).
* **Speed Suppression Engine:** Automatically suppresses critical alarms when the vehicle speed is $\le 10$ km/h (e.g., stationary at traffic lights) and flags violations at higher speeds.
* **Interactive Dashboard:** Real-time web UI built with NiceGUI featuring live video streaming, speed simulation controls, and risk analytics.
* **SQLite Persistence:** Asynchronous-safe event logging with write-throttling cooldowns to maintain high video frame rates.
---
## Tech Stack

* **Language:** Python 3.11
* **Computer Vision:** OpenCV, Ultralytics YOLOv8, MediaPipe Hands
* **UI Framework:** NiceGUI
* **Database:** SQLite3

[Click here to read my project blog](https://jyo-blogs.blogspot.com/2026/08/i2dms-intelligent-in-cabin-driver.html)
