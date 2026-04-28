import os

folder_name = "dataroot"
# files = [f for f in os.listdir(folder_name) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
files = []

for f in os.listdir(folder_name):
    if f.lower().endswith((".jpg", ".png", ".jpeg")):
        files.append(f)
        
print(f"🔄 Đang đổi tên {len(files)} ảnh trong {root_input}...")

for i, filename in enumerate(files):
    ext = filename.rsplit('.', 1)[-1]
    old_path = os.path.join(folder_name, filename)
    new_name = f"raw_{i+1:05d}.{ext}"
    new_path = os.path.join(folder_name, new_name)
    
    os.rename(old_path, new_path)

print("Doi ten thanh cong!")