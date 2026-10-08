#!/usr/bin/env python3
"""
Read and print sensor values from a Digit360 .mcap recording.
Usage: python3 read_data.py <bag_folder>
Example: python3 read_data.py $DATA_DIR/A/obj_01/tap/trial_000
"""

import sys
import os
import numpy as np
from mcap.reader import make_reader
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

def find_mcap_files(bag_path):
    files = sorted([
        os.path.join(bag_path, f)
        for f in os.listdir(bag_path)
        if f.endswith(".mcap")
    ])
    return files

def read_topic(mcap_files, topic, msg_type_str):
    msg_type = get_message(msg_type_str)
    messages = []
    for mcap_file in mcap_files:
        with open(mcap_file, "rb") as f:
            reader = make_reader(f)
            for schema, channel, message in reader.iter_messages(topics=[topic]):
                msg = deserialize_message(message.data, msg_type)
                messages.append((message.log_time, msg))
    return messages

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 read_data.py <bag_folder>")
        sys.exit(1)

    bag_path = os.path.expanduser(sys.argv[1])
    mcap_files = find_mcap_files(bag_path)
    print(f"\n{'='*60}")
    print(f"Reading: {bag_path}")
    print(f"Files: {len(mcap_files)}")
    print(f"{'='*60}\n")

    # ── IMU Raw ──────────────────────────────────────────────
    print("── IMU RAW (accelerometer + gyroscope) ──")
    try:
        msgs = read_topic(mcap_files, "/imu_raw_topic/index_0", "d360_msgs/msg/ImuRawD360")
        if msgs:
            values = [(m.x, m.y, m.z) for _, m in msgs]
            values = np.array(values)
            print(f"  Messages: {len(msgs)}")
            print(f"  X — mean: {values[:,0].mean():.4f}, std: {values[:,0].std():.4f}, min: {values[:,0].min():.4f}, max: {values[:,0].max():.4f}")
            print(f"  Y — mean: {values[:,1].mean():.4f}, std: {values[:,1].std():.4f}, min: {values[:,1].min():.4f}, max: {values[:,1].max():.4f}")
            print(f"  Z — mean: {values[:,2].mean():.4f}, std: {values[:,2].std():.4f}, min: {values[:,2].min():.4f}, max: {values[:,2].max():.4f}")
            print(f"  First message: ({values[0,0]:.3f}, {values[0,1]:.3f}, {values[0,2]:.3f})")
            accels = np.array(accels)
            gyros  = np.array(gyros)
            print(f"  Messages: {len(msgs)}")
            print(f"  Accel X — mean: {accels[:,0].mean():.4f}, std: {accels[:,0].std():.4f}, min: {accels[:,0].min():.4f}, max: {accels[:,0].max():.4f}")
            print(f"  Accel Y — mean: {accels[:,1].mean():.4f}, std: {accels[:,1].std():.4f}, min: {accels[:,1].min():.4f}, max: {accels[:,1].max():.4f}")
            print(f"  Accel Z — mean: {accels[:,2].mean():.4f}, std: {accels[:,2].std():.4f}, min: {accels[:,2].min():.4f}, max: {accels[:,2].max():.4f}")
            print(f"  Gyro  X — mean: {gyros[:,0].mean():.4f},  std: {gyros[:,0].std():.4f}")
            print(f"  Gyro  Y — mean: {gyros[:,1].mean():.4f},  std: {gyros[:,1].std():.4f}")
            print(f"  Gyro  Z — mean: {gyros[:,2].mean():.4f},  std: {gyros[:,2].std():.4f}")
            print(f"  First message: accel=({accels[0,0]:.3f}, {accels[0,1]:.3f}, {accels[0,2]:.3f})")
        else:
            print("  No messages found.")
    except Exception as e:
        print(f"  Error: {e}")

    print()

    # ── Pressure ─────────────────────────────────────────────
    print("── PRESSURE ──")
    try:
        msgs = read_topic(mcap_files, "/pressure_topic/index_0", "d360_msgs/msg/PressureD360")
        if msgs:
            print(f"  Messages: {len(msgs)}")
            # Print first message fields
            _, first = msgs[0]
            print(f"  First message fields: {[f for f in dir(first) if not f.startswith('_')]}")
            print(f"  Sample message: {first}")
        else:
            print("  No messages found.")
    except Exception as e:
        print(f"  Error: {e}")

    print()

    # ── Pressure AP ──────────────────────────────────────────
    print("── PRESSURE AP ──")
    try:
        msgs = read_topic(mcap_files, "/pressure_ap_topic/index_0", "d360_msgs/msg/PressureAPD360")
        if msgs:
            print(f"  Messages: {len(msgs)}")
            _, first = msgs[0]
            print(f"  Sample message: {first}")
        else:
            print("  No messages found.")
    except Exception as e:
        print(f"  Error: {e}")

    print()

    # ── Temperature / Humidity / Gas ─────────────────────────
    print("── GAS / HUMIDITY / TEMPERATURE ──")
    try:
        msgs = read_topic(mcap_files, "/ght_topic/index_0", "d360_msgs/msg/GasHTData")
        if msgs:
            print(f"  Messages: {len(msgs)}")
            for _, m in msgs:
                print(f"  Sample: {m}")
                break
        else:
            print("  No messages found.")
    except Exception as e:
        print(f"  Error: {e}")

    print()

    # ── Audio ────────────────────────────────────────────────
    print("── AUDIO (mic_0) ──")
    try:
        msgs = read_topic(mcap_files, "/mic_0/index_0", "d360_msgs/msg/AudioDataD360")
        if msgs:
            print(f"  Messages: {len(msgs)}")
            _, first = msgs[0]
            print(f"  Sample message fields: {[f for f in dir(first) if not f.startswith('_')]}")
            print(f"  Sample: {first}")
        else:
            print("  No messages found.")
    except Exception as e:
        print(f"  Error: {e}")

    print()
    print(f"{'='*60}")
    print("Done.")

if __name__ == "__main__":
    main()
