# tools/list_audio_devices.py
import pyaudio
import pprint

def list_devices():
    pa = pyaudio.PyAudio()

    print("\n=== Host APIs ===")
    for i in range(pa.get_host_api_count()):
        host = pa.get_host_api_info_by_index(i)
        print(f"[{i}] {host.get('name')} — devices: {host.get('deviceCount')}")

    print("\n=== Devices ===")
    for i in range(pa.get_device_count()):
        try:
            info = pa.get_device_info_by_index(i)
        except Exception:
            continue

        print(f"\n--- Device Index {i} ---")
        pprint.pprint(info)
        print("-------------------------")

    pa.terminate()


if __name__ == "__main__":
    list_devices()
