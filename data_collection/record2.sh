#!/bin/bash
# Digit360 Data Collection Script — Independent Trials Protocol
# Usage: [DATA_DIR=...] [ROS_WS=...] bash record2.sh <object_name> <sensor_id>
# Example: DATA_DIR=~/tactile_recordings bash record2.sh obj_01 A
#
# Protocol per object:
#   4 interactions × 5 trials × TRIAL_DURATION (2) seconds each
#   Interactions: tap, squeeze, slide, hold
#
# Environment:
#   DATA_DIR  where recordings are written (default: ~/tactile_recordings)
#   ROS_WS    ROS2 workspace containing install/setup.bash (default: ~/ros2_ws)
#
# Output: $DATA_DIR/<sensor_id>/<object_name>/<interaction>/trial_00X/

OBJECT_NAME=$1
SENSOR_ID=$2
DATA_DIR="${DATA_DIR:-$HOME/tactile_recordings}"
ROS_WS="${ROS_WS:-$HOME/ros2_ws}"
DATA_ROOT="$DATA_DIR"
TRIAL_DURATION=2        # recording duration in seconds 
N_TRIALS=5
PAUSE_BETWEEN_TRIALS=3  # seconds to reposition object between trials
PAUSE_BETWEEN_ACTIONS=5 # seconds between interactions

if [ -z "$OBJECT_NAME" ] || [ -z "$SENSOR_ID" ]; then
    echo "ERROR: Missing arguments."
    echo "Usage: bash record2.sh <object_name> <sensor_id>"
    echo "Example: bash record2.sh obj_01 A"
    exit 1
fi

OBJ_DIR="$DATA_ROOT/$SENSOR_ID/$OBJECT_NAME"
mkdir -p "$OBJ_DIR"

echo "================================================"
echo "  Digit360 Data Collection — Independent Trials"
echo "  Object:    $OBJECT_NAME"
echo "  Sensor ID: $SENSOR_ID"
echo "  Trials:    $N_TRIALS × ${TRIAL_DURATION}s per interaction"
echo "  Output:    $OBJ_DIR"
echo "================================================"
echo ""

source /opt/ros/jazzy/setup.bash
source "$ROS_WS/install/setup.bash"

# ── Launch sensor node ONCE ──────────────────────────
echo "[SETUP] Starting Digit360 sensor node..."
sleep 3 
ros2 launch d360 d360_min_launch.py > /tmp/d360_launch.log 2>&1 &
LAUNCH_PID=$!

# Wait for topics
echo "[SETUP] Waiting for sensor topics..."
TIMEOUT=30
ELAPSED=0
while true; do
    TOPICS=$(ros2 topic list 2>/dev/null)
    if echo "$TOPICS" | grep -q "/image_raw/index_0" && \
       echo "$TOPICS" | grep -q "/imu_raw_topic/index_0" && \
       echo "$TOPICS" | grep -q "/pressure_topic/index_0"; then
        echo "[SETUP] Sensor ready!"
        sleep 8  # extra wait for camera to stabilize
        break
    fi
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "ERROR: Sensor not found after ${TIMEOUT}s. Check connection."
        kill $LAUNCH_PID 2>/dev/null
        exit 1
    fi
    sleep 1
    ELAPSED=$((ELAPSED + 1))
    echo "         Waiting... ($ELAPSED/${TIMEOUT}s)"
done

echo ""

# ──  countdown ────────────────────────────────
countdown() {
    local seconds=$1
    local msg=${2:-"Next trial in"}
    while [ $seconds -gt 0 ]; do
        echo -ne "    $msg $seconds seconds...\r"
        sleep 1
        seconds=$((seconds - 1))
    done
    echo -ne "                                        \r"
}

# ──  record one trial ─────────────────────────
record_trial() {
    local output=$1
    local duration=$2

    if [ -d "$output" ]; then
        echo "    WARNING: $(basename $output) already exists — skipping."
        return 0
    fi
    sleep 0.5
    ros2 bag record -a -o "$output" --max-bag-duration $duration &
    BAG_PID=$!
    sleep $((duration + 2))
    kill $BAG_PID 2>/dev/null
    wait $BAG_PID 2>/dev/null
    echo "    ✅ Saved: $output"
}

# ── run all trials for one interaction ────────
run_interaction() {
    local interaction=$1
    local description=$2
    local out_base="$OBJ_DIR/$interaction"

    echo "────────────────────────────────────────────────"
    echo "  INTERACTION: ${interaction^^} — $description"
    echo "  $N_TRIALS trials × ${TRIAL_DURATION} seconds"
    echo "────────────────────────────────────────────────"
    echo ""

    for i in $(seq 0 $((N_TRIALS - 1))); do
        TRIAL_NAME="trial_$(printf '%03d' $i)"
        TRIAL_PATH="$out_base/$TRIAL_NAME"

        if [ -d "$TRIAL_PATH" ]; then
            echo "    Skipping $TRIAL_NAME — already exists."
            continue
        fi
        read -p "  [Trial $((i+1))/$N_TRIALS] Press ENTER to record $TRIAL_NAME — ${interaction^^}..."
        echo "  *** RECORDING ***"
        record_trial "$TRIAL_PATH" $TRIAL_DURATION

        if [ $i -lt $((N_TRIALS - 1)) ]; then
            countdown $PAUSE_BETWEEN_TRIALS "  Reposition object —"
        fi
    done

    echo ""
    echo "  ✅ $interaction complete — $N_TRIALS trials saved."
    echo ""
}

# ── Protocol ─────────────────────────────────────────
echo "┌──────────────────────────────────────────────┐"
echo "│  PROTOCOL FOR: $OBJECT_NAME (Sensor $SENSOR_ID)"
echo "│  tap     → $N_TRIALS trials × ${TRIAL_DURATION}s                     │"
echo "│  squeeze → $N_TRIALS trials × ${TRIAL_DURATION}s                     │"
echo "│  slide   → $N_TRIALS trials × ${TRIAL_DURATION}s                     │"
echo "│  hold    → $N_TRIALS trials × ${TRIAL_DURATION}s                     │"
echo "└──────────────────────────────────────────────┘"
echo ""

run_interaction "tap"     "Repeatedly tap the object on the sensor"
countdown $PAUSE_BETWEEN_ACTIONS "Next interaction in"

run_interaction "squeeze" "Press and hold the object against the sensor"
countdown $PAUSE_BETWEEN_ACTIONS "Next interaction in"

run_interaction "slide"   "Slowly slide the object across the sensor surface"
countdown $PAUSE_BETWEEN_ACTIONS "Next interaction in"

run_interaction "hold"    "Hold the object still against the sensor, no movement"

# ── Done ─────────────────────────────────────────────
echo ""
echo "================================================"
echo "  ALL DONE! Object: $OBJECT_NAME | Sensor: $SENSOR_ID"
echo "  Saved to: $OBJ_DIR"
echo "================================================"
echo ""
echo "  Structure:"
find "$OBJ_DIR" -type d | sort
echo ""

# Stop sensor node
kill $LAUNCH_PID 2>/dev/null
wait $LAUNCH_PID 2>/dev/null

echo "  Sensor node stopped."
echo "  Run next object: bash record2.sh <next_object> $SENSOR_ID"
echo ""
