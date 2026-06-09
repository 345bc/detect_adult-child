import os

def generate_yaml():
    # Get the project root directory
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    datasets_dir = os.path.join(project_dir, "datasets")
    
    # Path format should use forward slashes for YOLO compatibility
    datasets_dir_formatted = datasets_dir.replace("\\", "/")
    
    yaml_content = f"""path: {datasets_dir_formatted} # dataset root dir (absolute path)
train: train/images
val: val/images
test: test/images

nc: 2
# Classes
names:
  0: Adult
  1: Child
"""
    
    output_path = os.path.join(project_dir, "data.yaml")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
        
    print(f"✅ Đã tạo thành công data.yaml tại: {output_path}")
    print("Nội dung file:")
    print(yaml_content)

if __name__ == "__main__":
    generate_yaml()
