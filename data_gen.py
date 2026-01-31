import numpy as np
import pandas as pd
from math import sin
from datetime import datetime, timedelta


# -------------------------------
# Helper functions
# -------------------------------

def gaussian_noise(mu=0.0, sigma=1.0):
    return np.random.normal(mu, sigma)

def ambient_temperature(time_s):
    # Slow sinusoidal drift + small noise
    return 25.0 + 2.0 * sin(time_s / 3600.0) + gaussian_noise(0, 0.2)

def rf_power_schedule(time_s):
    # Simple step change to test compression
    return 5.0 if time_s < 1800 else 10.0

def rf_frequency_schedule(time_s):
    # Fixed for Stage 0–3
    return 3e9

def rf_output_state(time_s):
    # ON for 1000s, OFF for 200s (thermal cycling)
    return True if (time_s % 1200) < 1000 else False

# -------------------------------
# Simulation parameters
# -------------------------------

dt = 1.0                       # seconds
num_steps = 3600               # 1 hour
start_time = datetime.now()

# Reference constants
T_REF = 25.0                   # °C
V_SUPPLY = 28.0                # Volts

# -------------------------------
# Initial states
# -------------------------------

time_s = 0.0
aging_factor = 0.0
pa_temperature = T_REF
freq_drift_state = 0.0

# -------------------------------
# Storage
# -------------------------------

rows = []

# -------------------------------
# MAIN SIMULATION LOOP
# -------------------------------

for step in range(num_steps):

    # ---------------------------
    # STAGE 0 — Exogenous Inputs
    # ---------------------------
    time_s += dt
    ts = start_time + timedelta(seconds=time_s)

    aging_factor += 1e-7 * dt
    ambient_temp = ambient_temperature(time_s)

    rf_power_setpoint = rf_power_schedule(time_s)
    rf_freq_setpoint = rf_frequency_schedule(time_s)
    rf_on = rf_output_state(time_s)

    # ---------------------------
    # STAGE 1 — PA Supply Current
    # ---------------------------
    I_IDLE = 0.5
    K_POWER = 0.15
    K_AGING = 2.0

    if rf_on:
        pa_supply_current = (
            I_IDLE
            + K_POWER * rf_power_setpoint
            + K_AGING * aging_factor
            + gaussian_noise(0, 0.05)
        )
    else:
        pa_supply_current = 0.0

    # ---------------------------
    # STAGE 2 — PA Temperature
    # ---------------------------
    ALPHA = 0.02
    BETA = 0.01

    power_dissipated = V_SUPPLY * pa_supply_current - rf_power_setpoint

    pa_temperature = (
        pa_temperature
        + ALPHA * power_dissipated
        - BETA * (pa_temperature - ambient_temp)
        + gaussian_noise(0, 0.1)
    )

    # ---------------------------
    # STAGE 3A — RF Output Power
    # ---------------------------
    P_LINEAR_END = 10.0
    K_TEMP_COMP = 0.02
    K_AGING_COMP = 1.5
    MAX_COMPRESSION = 6.0

    compression = max(
        0.0,
        (rf_power_setpoint - P_LINEAR_END)
        + K_TEMP_COMP * max(0.0, pa_temperature - T_REF)
        + K_AGING_COMP * aging_factor
    )

    compression = min(compression, MAX_COMPRESSION)

    measured_rf_output = (
        rf_power_setpoint
        - compression
        + gaussian_noise(0, 0.2)
    )

    # ---------------------------
    # STAGE 3B — Internal Frequency Error
    # ---------------------------
    K_TEMP_FREQ = 1.0
    DRIFT_NOISE = 0.05

    freq_drift_state += gaussian_noise(0, DRIFT_NOISE)

    freq_error_internal = (
        K_TEMP_FREQ * (pa_temperature - T_REF)
        + freq_drift_state
        + gaussian_noise(0, 1.0)
    )

    # ---------------------------
    # STAGE 3C — External Frequency Error
    # ---------------------------
    K_EXT_TEMP_FREQ = 0.2

    freq_error_external = (
        K_EXT_TEMP_FREQ * (pa_temperature - T_REF)
        + gaussian_noise(0, 0.5)
    )

    # ---------------------------
    # Store telemetry row
    # ---------------------------
    rows.append({
        "ts": ts,
        "rf_frequency_setpoint_hz": rf_freq_setpoint,
        "rf_power_setpoint_dbm": rf_power_setpoint,
        "rf_output_state": rf_on,
        "pa_supply_current_a": pa_supply_current,
        "pa_temperature_c": pa_temperature,
        "measured_rf_output_dbm": measured_rf_output,
        "freq_error_internal_hz": freq_error_internal,
        "freq_error_external_hz": freq_error_external,
        "data_source": "synthetic_physics"
    })

# -------------------------------
# Create DataFrame
# -------------------------------

df = pd.DataFrame(rows)

print(df.head())
print("\nDataset shape:", df.shape)

# -------------------------------
# Write to CSV
# -------------------------------

df.to_csv("data_gen.csv", index=False)