import os
import numpy as np
import soundfile as sf

os.makedirs("data", exist_ok=True)

# Generate 5 synthetic audio samples
for i in range(5):
    sr = 16000  # 16 kHz sample rate
    duration = 2  # 2 seconds
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)

    # Random waveform: sine + noise
    freq = np.random.randint(100, 1000)  # random frequency
    waveform = 0.5 * np.sin(2 * np.pi * freq * t) + 0.05 * np.random.randn(len(t))

    file_path = f"data/sample_{i}.wav"
    sf.write(file_path, waveform, sr)
    print(f"Generated {file_path}")
print("✅ Synthetic audio data generation complete.")