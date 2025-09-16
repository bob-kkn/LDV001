# Panorama PII Blurring (Faces & Plates)

파노라마/일반 이미지·비디오에서 **얼굴/번호판**을 자동 검출해 블러(또는 모자이크) 처리하는 도구입니다.  
CLI와 **tkinter GUI** 두 가지 실행 방식을 제공합니다.

- 엔진: OpenCV + (선택) Ultralytics YOLO
- 입력: 단일 이미지/동영상 또는 폴더(재귀)
- 출력: 원본 확장자 유지, `_blurred` 후미 또는 동일 폴더 구조 복사
- 검출기: **YOLO**(GPU/CPU) 또는 **OpenCV Cascade**(CPU)
- 최적화: 수평 타일링(+phase), 상하 polar band 보강, NMS/Soft-NMS, (선택) 향상·업스케일, 타일 배치 추론
- 폴더 모드 멀티프로세스 지원(CLI)
- GUI: 메인(입력/출력/실행/중지) + 설정창(상세 옵션)

---

## 설치

### 1) Python & 패키지
- Python **3.9+** 권장
- 필수:
  ```bash
  pip install opencv-python numpy
YOLO 사용 시(선택):

bash
코드 복사
pip install ultralytics torch torchvision --upgrade
CUDA 사용하려면 CUDA 지원 PyTorch 필요

2) 모델(가중치) 파일
bash
코드 복사
./models/yolov8x-face-lindevs.pt
./models/best.pt
파일 구성
bash
코드 복사
pano_blur.py        # 엔진(검출·블러·입출력·CLI)
pano_blur_gui.py    # GUI (tkinter)
빠른 시작 (권장 기본값)
bash
코드 복사
python pano_blur.py \
  --input ./input \
  --out-dir ./output \
  --yolo-face ./models/yolov8x-face-lindevs.pt \
  --yolo-plate ./models/best.pt \
  --fast \
  --tiles 10
GUI 실행
bash
코드 복사
python pano_blur_gui.py
최초 실행 시 기본값:

Input: ./input

Output: ./output

YOLO Face: ./models/yolov8x-face-lindevs.pt

YOLO Plate: ./models/best.pt

FAST preset: ON

Tiles: 10

설정 저장 위치: ~/.pano_blur_gui_settings.json

(선택) GUI 기본값을 커맨드라인으로 오버라이드
bash
코드 복사
python pano_blur_gui.py \
  --input ./input \
  --out-dir ./output \
  --yolo-face ./models/yolov8x-face-lindevs.pt \
  --yolo-plate ./models/best.pt \
  --fast --tiles 10
성능 & GPU
YOLO 사용 시(가중치 지정) Ultralytics가 가능하면 GPU(CUDA/MPS) 사용

Cascade/블러/입출력은 CPU

단일 GPU에서 폴더 멀티프로세스는 속도 저하 가능 → YOLO 사용 시 jobs=1 권장

문제 해결
ImportError: ultralytics/torch → 해당 패키지 설치

동영상 쓰기 실패 → FOURCC 확인(mp4v, XVID 등)

속도 개선 → --fast, --video-step, --tiles, --yolo-imgsz 조절, YOLO+GPU 사용

yaml
코드 복사

---

## 저장 팁 (Windows)
메모장 대신 VS Code/Notepad++로 **UTF-8** 인코딩으로 저장하세요.  
폴더 구조 예:
your_project/
├─ pano_blur.py
├─ pano_blur_gui.py
├─ README.md
└─ models/
├─ yolov8x-face-lindevs.pt
└─ best.pt