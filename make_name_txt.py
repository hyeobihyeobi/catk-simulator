import os

# scene npy 파일들이 있는 디렉토리 경로
directory = "/workspace/catk-simulator/train_preprocessed_path/tl_status"   # ← 여기 수정

# 출력할 txt 파일 이름
output_txt = "/workspace/catk-simulator/name_txt/name.txt"

scene_ids = []

for filename in os.listdir(directory):
    if filename.endswith(".npy"):
        scene_id = os.path.splitext(filename)[0]  # .npy 제거
        scene_ids.append(scene_id)

# 정렬(Optional)
scene_ids = sorted(scene_ids, key=lambda x: int(x))

# txt 파일로 저장
with open(output_txt, "w") as f:
    for sid in scene_ids:
        f.write(f"{sid}\n")

print(f"총 {len(scene_ids)}개의 scene_id를 '{output_txt}'에 저장했습니다.")
