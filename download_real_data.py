import os
import urllib.request
import ssl

def download_dataset():
    # Setup directories
    base_dir = "data"
    heart_dir = os.path.join(base_dir, "Heart")
    lung_dir = os.path.join(base_dir, "Lung")
    os.makedirs(heart_dir, exist_ok=True)
    os.makedirs(lung_dir, exist_ok=True)
    
    # Disable SSL verification issues if any
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    print("Downloading REAL clinical Heart Sound recordings (PhysioNet PCG)...")
    # PhysioNet 2016 Heart Sound Challenge recordings (a0001 to a0015)
    for i in range(1, 16):
        file_name = f"a{i:04d}.wav"
        url = f"https://physionet.org/files/challenge-2016/1.0.0/training-a/{file_name}"
        save_path = os.path.join(heart_dir, f"heart_real_{i}.wav")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, context=ctx, timeout=10) as response, open(save_path, 'wb') as out_file:
                out_file.write(response.read())
            print(f"  [+] Saved {save_path}")
        except Exception as e:
            print(f"  [-] Failed to download {url}: {e}")

    print("\nDownloading REAL clinical Lung / Breath Sound recordings...")
    # Real breath sound recordings from open respiratory benchmark
    lung_files = [
        "01_male_23_BQuyen.wav", "02_male_22_PTuan.wav", "03_male_21_BDuong.wav",
        "04_female_21_LAnh.wav", "05_male_21_NLinh.wav", "06_female_22_NTam.wav",
        "07_female_22_NThuy.wav", "08_male_22_VThinh.wav", "09_male_22_TViet.wav",
        "10_male_22_PVuong.wav", "11_female_21_MDung.wav", "12_male_21_HTrang.wav",
        "13_female_21_TThuy.wav", "14_male_22_HThang.wav", "15_male_21_HPhuong.wav"
    ]
    
    for i, fname in enumerate(lung_files, 1):
        url = f"https://raw.githubusercontent.com/PTITLab/Multitask-Breath-Sound/master/Breath-Data/{fname}"
        save_path = os.path.join(lung_dir, f"lung_real_{i}.wav")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, context=ctx, timeout=10) as response, open(save_path, 'wb') as out_file:
                out_file.write(response.read())
            print(f"  [+] Saved {save_path}")
        except Exception as e:
            print(f"  [-] Failed to download {url}: {e}")

    print("\nDownload complete! Real biomedical audio dataset is ready.")

if __name__ == "__main__":
    download_dataset()
