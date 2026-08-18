"""
I2DMS - Speed Tracker Module
Demonstrates client-side raw HTML5 GPS capture + server-side anti-spoofing validation.
"""

from math import radians, cos, sin, asin, sqrt
import time
from nicegui import ui

class SpeedValidator:
    def __init__(self):
        self.last_lat = None
        self.last_lon = None
        self.last_time = None
        self.last_speed_kmh = 0.0

    def validate_and_calculate(self, lat: float, lon: float, accuracy: float) -> tuple[float, str]:
        now = time.time()

        # 1. Reject inaccurate GPS fixes (> 25 meters error margin)
        if accuracy > 25.0:
            return self.last_speed_kmh, "GPS Signal Weak (Error > 25m)"

        # First reading initialization
        if self.last_lat is None or self.last_time is None:
            self.last_lat = lat
            self.last_lon = lon
            self.last_time = now
            return 0.0, "GPS Signal Acquired"

        dt = now - self.last_time
        if dt < 0.2:  # Ignore duplicate or rapid duplicate frames
            return self.last_speed_kmh, "Active Tracking"

        # 2. Server-Side Haversine Distance Formula
        r = 6371000.0  # Earth's radius in meters
        dlat = radians(lat - self.last_lat)
        dlon = radians(lon - self.last_lon)
        
        a = (sin(dlat / 2) ** 2 + 
             cos(radians(self.last_lat)) * cos(radians(lat)) * sin(dlon / 2) ** 2)
        dist_meters = 2 * r * asin(sqrt(a))
        
        speed_ms = dist_meters / dt
        calculated_speed_kmh = speed_ms * 3.6

        # 3. Acceleration Plausibility Check (Flag jumps > 15 m/s² (~54 km/h per second))
        accel = abs(speed_ms - (self.last_speed_kmh / 3.6)) / dt
        if accel > 15.0 and dist_meters > 10.0:
            return self.last_speed_kmh, "Spoof Warning: Impossible Acceleration Jump"

        # Update cache
        self.last_lat = lat
        self.last_lon = lon
        self.last_time = now
        self.last_speed_kmh = max(0.0, calculated_speed_kmh)

        return self.last_speed_kmh, "Validated Server-Side"


validator = SpeedValidator()

# NICEGUI 

@ui.page('/')
def main_page():
    ui.page_title("Live Speed Telemetry Test")
    
    with ui.card().classes("w-full max-w-md mx-auto mt-6 p-6 text-center shadow-lg bg-slate-900 text-white rounded-2xl"):
        ui.label("VEHICLE SPEED TRACKER").classes("text-xs font-semibold tracking-wider text-slate-400")
        
        speed_label = ui.label("0.0").classes("text-6xl font-extrabold text-sky-400 my-4")
        ui.label("km/h").classes("text-sm text-slate-400 font-medium")

        status_badge = ui.label("Waiting for GPS...").classes("text-xs mt-4 py-1 px-3 rounded-full bg-amber-500/20 text-amber-300 inline-block")

    with ui.card().classes("w-full max-w-md mx-auto mt-4 p-4 shadow-md bg-slate-800 text-white rounded-xl"):
        ui.label("Raw Telemetry Diagnostics").classes("font-semibold text-sm mb-3 text-slate-300")
        
        with ui.grid(columns=2).classes("w-full text-xs gap-2 text-slate-400"):
            ui.label("Latitude:")
            lat_label = ui.label("--").classes("text-slate-200 font-mono")
            
            ui.label("Longitude:")
            lon_label = ui.label("--").classes("text-slate-200 font-mono")
            
            ui.label("Accuracy:")
            acc_label = ui.label("--").classes("text-slate-200 font-mono")
            
            ui.label("Validation Engine:")
            engine_label = ui.label("Server-Side (Anti-Spoof)").classes("text-emerald-400 font-medium")

    # Ingest Raw Payload Emitted by Browser (No JS Calculations)
    def handle_raw_gps(e):
        data = e.args
        lat = float(data['lat'])
        lon = float(data['lon'])
        acc = float(data['acc'])

        # Process through Python validator
        speed, status_msg = validator.validate_and_calculate(lat, lon, acc)

        # Update UI Elements
        speed_label.set_text(f"{speed:.1f}")
        lat_label.set_text(f"{lat:.5f}")
        lon_label.set_text(f"{lon:.5f}")
        acc_label.set_text(f"±{acc:.1f} m")
        status_badge.set_text(status_msg)

        if "Spoof" in status_msg or "Weak" in status_msg:
            status_badge.classes(replace="bg-red-500/20 text-red-300")
        else:
            status_badge.classes(replace="bg-emerald-500/20 text-emerald-300")

    ui.on('raw_gps_data', handle_raw_gps)

    # Browser JavaScript: Streams RAW metrics only. 
    gps_js = """
    if ("geolocation" in navigator) {
        navigator.geolocation.watchPosition(
            (pos) => {
                emitEvent('raw_gps_data', {
                    'lat': pos.coords.latitude,
                    'lon': pos.coords.longitude,
                    'acc': pos.coords.accuracy
                });
            },
            (err) => {},
            {
                enableHighAccuracy: true,
                maximumAge: 1000,
                timeout: 5000
            }
        );
    }
    """
    ui.run_javascript(gps_js)

ui.run(title="Speed Tracker Module", port=8081, reload=False)
