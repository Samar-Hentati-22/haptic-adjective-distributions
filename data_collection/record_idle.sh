#!/bin/bash
# Digit360 Idle Recording Script
# Run this ONCE at the start of each sensor session.
# Usage: [DATA_DIR=...] [ROS_WS=...] bash record_idle.sh <sensor_id>
# Example: DATA_DIR=~/tactile_recordings bash record_idle.sh A
#
# Environment:
#   DATA_DIR  where recordings are written (default: ~/tactile_recordings)
#   ROS_WS    ROS2 workspace containing install/setup.bash (default: ~/ros2_ws)
#
# Data saved to: $DATA_DIR/<sensor_id>/idle/

SENSOR_ID=$1
DATA_DIR="${DATA_DIR:-$HOME/tactile_recordings}"
ROS_WS="${ROS_WS:-$HOME/ros2_ws}"
RAW_DATA="$DATA_DIR"
IDLE_DURATION=5

if [ -z "$SENSOR_ID" ]; then
    echo "ERROR: Missing sensor ID."
    echo "Usage: bash record_idle.sh <sensor_id>"
    echo "Example: bash record_idle.sh A"
    exit 1
fi

OUTPUT="$RAW_DATA/$SENSOR_ID/idle"
mkdir -p "$RAW_DATA/$SENSOR_ID"

if [ -d "$OUTPUT" ]; then
    echo "WARNING: Idle recording for sensor '$SENSOR_ID' already exists."
    read -p "Overwrite? (y/n): " confirm
    if [ "$confirm" != "y" ]; then
        echo "Aborted."
        exit 0
    fi
    rm -rf "$OUTPUT"
fi

echo "================================================"
echo "  Digit360 Idle Recording"
echo "  Sensor ID: $SENSOR_ID"
echo "  Duration:  $IDLE_DURATION seconds"
echo "  Output:    $OUTPUT"
echo "================================================"
echo ""

# Source ROS2
source /opt/ros/jazzy/setup.bash
source "$ROS_WS/install/setup.bash"

# Launch sensor node
echo "[SETUP] Starting Digit360 sensor node..."
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
echo "────────────────────────────────────────────────"
echo "  IDLE RECORDING"
echo "  Hold the sensor in the air — NO CONTACT."
echo "────────────────────────────────────────────────"
read -p "  Press ENTER when ready..."
echo ""
echo "  *** RECORDING STARTED ***"
echo ""

ros2 bag record -a -o "$OUTPUT" --max-bag-duration $IDLE_DURATION > /dev/null 2>&1 &
BAG_PID=$!
sleep $((IDLE_DURATION + 2))
kill $BAG_PID 2>/dev/null
wait $BAG_PID 2>/dev/null

echo "  ✅ Idle recording saved to: $OUTPUT"
echo ""

# Stop sensor node
kill $LAUNCH_PID 2>/dev/null
wait $LAUNCH_PID 2>/dev/null

echo "  Sensor node stopped."
echo "  You can now start recording objects:"
echo "  bash record2.sh <object_name> $SENSOR_ID"
echo ""
