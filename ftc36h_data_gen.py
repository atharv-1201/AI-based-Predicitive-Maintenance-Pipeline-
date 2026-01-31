import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ============================================================
# CONFIGURATION
# ============================================================

DT = 1.0                              
HOURS = 36
NUM_STEPS = int(HOURS * 3600)
RF_FREQ_HZ = 3e9

OUTPUT_CSV = "ftc36h.csv"

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# Trajectory targets (minimums)
MIN_INCI = 8
MIN_RECV = 8
MIN_TERM = 12

# ============================================================
# FAILURE MODE ASSIGNMENT
# ============================================================

FAILURE_MODES = (
    ["normal"] * 40 +
    ["incipient"] * MIN_INCI +
    ["recovering"] * MIN_RECV +
    ["terminal"] * MIN_TERM
)

rng.shuffle(FAILURE_MODES)

# ============================================================
# PHYSICS CONSTANTS
# ============================================================

T_REF = 25.0
V_SUPPLY = 28.0

# FTC parameters
FTC_TIMER_THRESHOLD = {
    "terminal": 1200,     
    "incipient": 300,
    "recovering": 300
}

FTC_AMP_CAP = {
    "normal": 1.0,
    "incipient": 1.25,
    "recovering": 1.7,
    "terminal": 3.0
}

# Degradation parameters

BASE_ACLR_DEG = 2.5e-4
BASE_EVM_DEG  = 1.5e-4


FTC_THERMAL_GAIN = 0.04   # °C / step (terminal only)
EPS_ACLR = 0.02           # dB / step
EPS_EVM  = 0.02           # % / step

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def rf_power_schedule(t):
    if t < 8 * 3600:
        return 6.0
    elif t < 20 * 3600:
        return 10.0
    else:
        return 8.0

def rf_output_state(t):
    return True if (t % 1800) < 1500 else False

def ambient_temperature(t):
    return 25 + 3 * np.sin(t / 7200) + rng.normal(0, 0.3)

# ============================================================
# SIMULATION
# ============================================================

rows = []
start_time = datetime.now()

for traj_id, FAILURE_MODE in enumerate(FAILURE_MODES):

    # Initial states
    time_s = 0.0
    aging_factor = 0.0
    pa_temp = T_REF
    freq_drift = 0.0

    prev_aclr = 45.0
    prev_evm = 1.0

    # Degradation state variables 
    aclr_deg_state = 0.0
    evm_deg_state  = 0.0
    
    if FAILURE_MODE == "terminal":
        aclr_deg_rate = rng.uniform(0.8, 1.4)
        evm_deg_rate  = rng.uniform(0.7, 1.5)
    else:
        aclr_deg_rate = 1.0
        evm_deg_rate  = 1.0


    ftc_timer = 0
    ftc_active = False
    ftc_amp = 1.0

    for step in range(NUM_STEPS):
        time_s += DT
        ts = start_time + timedelta(seconds=time_s)

        # -------------------------------
        # STAGE 0 — EXOGENOUS
        # -------------------------------
        aging_factor += 1e-7
        ambient = ambient_temperature(time_s)
        rf_power = rf_power_schedule(time_s)
        rf_on = rf_output_state(time_s)

        # -------------------------------
        # STAGE 1 — CURRENT
        # -------------------------------
        if rf_on:
            pa_current = (
                0.5 +
                0.15 * rf_power +
                2.0 * aging_factor +
                rng.normal(0, 0.05)
            )
        else:
            pa_current = 0.0

        # -------------------------------
        # STAGE 2 — TEMPERATURE
        # -------------------------------
        pa_temp += (
            0.02 * (V_SUPPLY * pa_current - rf_power)
            - 0.01 * (pa_temp - ambient)
            + rng.normal(0, 0.1)
        )

        # FTC thermal runaway (terminal only)
        if FAILURE_MODE == "terminal" and ftc_active:
            pa_temp += FTC_THERMAL_GAIN * ftc_amp

        # -------------------------------
        # STRESS & FTC LOGIC
        # -------------------------------
        stress_condition = (rf_power >= 9.5 and pa_temp > 60)

        if stress_condition:
            ftc_timer += 1
        else:
            ftc_timer = max(0, ftc_timer - 1)

        if (
            FAILURE_MODE != "normal"
            and ftc_timer >= FTC_TIMER_THRESHOLD[FAILURE_MODE]
        ):
            ftc_active = True

        # -------------------------------------------------
        # TERMINAL DEGRADATION STATE EVOLUTION (NEW)
        # -------------------------------------------------

        if FAILURE_MODE == "terminal" and ftc_active:
            # Slow, irreversible degradation accumulation
            aclr_deg_state += rng.uniform(0.005, 0.015)
            evm_deg_state  += rng.uniform(0.01, 0.03)


        if ftc_active:
            ftc_amp = min(
                ftc_amp + 0.001,
                FTC_AMP_CAP[FAILURE_MODE]
            )
        else:
            ftc_amp = max(1.0, ftc_amp - 0.001)

        
            # DEGRADATION STATE EVOLUTION (NEW, REQUIRED)


        if FAILURE_MODE == "terminal" and ftc_active:
            aclr_deg_state += BASE_ACLR_DEG * ftc_amp * aclr_deg_rate
            evm_deg_state  += BASE_EVM_DEG  * ftc_amp * evm_deg_rate


        # -------------------------------
        # STAGE 3 — RF METRICS
        # -------------------------------
        compression = max(
            0,
            (rf_power - 10)
            + 0.02 * max(0, pa_temp - T_REF)
            + 1.5 * aging_factor
        )

        measured_rf = rf_power - compression + rng.normal(0, 0.2)

        freq_drift += rng.normal(0, 0.05)
        freq_error_int = (
            (pa_temp - T_REF)
            + freq_drift
            + rng.normal(0, 1.0)
        )

        freq_error_ext = (
            0.2 * (pa_temp - T_REF)
            + rng.normal(0, 0.5)
        )

        # # -------------------------------
        # # STAGE 4 — ACLR
        # # -------------------------------
        # aclr = (
        #     prev_aclr
        #     - ftc_amp * (0.08 * max(0, rf_power - 5))
        #     - ftc_amp * (0.05 * max(0, pa_temp - 40))
        #     + rng.normal(0, 0.4)
        # )

        # if FAILURE_MODE == "terminal" and ftc_active:
        #     eps_aclr = rng.uniform(0.01, 0.03)  # dB per step
        #     aclr = min(aclr, prev_aclr - eps_aclr)

        # aclr = np.clip(aclr, 0, 100)


        # -------------------------------------------------
        # STAGE 4 — ACLR (SOFT DEGRADATION WITH STATE)
        # -------------------------------------------------

        # Base physical ACLR response
        aclr_physical = (
            prev_aclr
            - 0.06 * max(0, rf_power - 5)
            - 0.04 * max(0, pa_temp - 40)
            + rng.normal(0, 0.3)
        )

        # Terminal degradation bias (NEW)
        if FAILURE_MODE == "terminal" and ftc_active:
            aclr = aclr_physical - aclr_deg_state
        else:
            aclr = aclr_physical

        # Soft bounding only
        aclr = min(aclr, 100)


        # -------------------------------
        # STAGE 5 — EVM
        # -------------------------------
        # evm = (
        #     prev_evm
        #     + 0.12 * max(0, 45 - aclr)
        #     + 0.03 * max(0, pa_temp - 40)
        #     + 0.4 * aging_factor
        #     - 0.1 * max(0, prev_evm - 0.5)
        #     + rng.normal(0, 0.08)
        # )

        # if FAILURE_MODE == "terminal" and ftc_active:
        #     eps_evm = rng.uniform(0.05, 0.15)  # % per step
        #     evm = prev_evm + eps_evm

        # # Soft cap, NOT hard pin
        # EVM_MAX = 25.0
        # evm = np.clip(evm, 0, EVM_MAX)



        # -------------------------------
        # STAGE 5 — EVM (ASYMPTOTIC)
        # -------------------------------

        # Base physical contribution 
        evm_physical = (
            0.12 * max(0, 45 - aclr)
            + 0.03 * max(0, pa_temp - 40)
            + 0.4 * aging_factor
        )

        # Degradation state contribution
        evm_increment = evm_physical + evm_deg_state

        # Asymptotic slowdown 
        EVM_SOFT_SCALE = 25.0   
        asymptotic_factor = 1.0 / (1.0 + prev_evm / EVM_SOFT_SCALE)

        # Increment smoothly
        evm = prev_evm + evm_increment * asymptotic_factor

        # Noise s
        evm += rng.normal(0, 0.08)

        # Terminal FTC guarantees directionality (NO PINNING)
        if FAILURE_MODE == "terminal" and ftc_active:
            evm = max(evm, prev_evm)

        prev_aclr = aclr
        prev_evm = evm

       
        rows.append({
            "ts": ts,
            "rf_frequency_setpoint_hz": RF_FREQ_HZ,
            "trajectory_id": traj_id,
            "failure_mode": FAILURE_MODE,
            "rf_power_setpoint_dbm": rf_power,
            "rf_output_state": rf_on,
            "pa_supply_current_a": pa_current,
            "pa_temperature_c": pa_temp,
            "measured_rf_output_dbm": measured_rf,
            "freq_error_internal_hz": freq_error_int,
            "freq_error_external_hz": freq_error_ext,
            "aclr_db": aclr,
            "rms_evm_percent": evm,
            "aging_factor": aging_factor,
            "ftc_active": ftc_active,
            "data_source": "synthetic_physics"
        })



df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)

print("36h FTC dataset generated")
print(df["failure_mode"].value_counts())
print("Saved to:", OUTPUT_CSV)
