"""
FastAPI 실시간 침입 감지 서버 (8주차 - threshold 개선 버전)

- 라즈베리파이로부터 이미지를 POST로 받음
- YOLO + DeepFace로 등록/미등록 인물 판별
- MySQL 저장 + Telegram + Email 이중 알림
- 이벤트별 알림 쿨다운 지원 (현재 실험 설정: 0초, 제한 없음)
- threading 기반 비동기 알림
- .env 환경변수 분리
- 등록/미등록 구분을 위해 2단계 threshold 적용

판별 기준:
distance < 0.30       -> Registered
0.30 <= distance < 0.47 -> Unidentified
distance >= 0.47      -> Unknown
"""

import os
import time
import cv2
import numpy as np
import pymysql
import requests
import smtplib
import threading

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from ultralytics import YOLO
from deepface import DeepFace
from dotenv import load_dotenv


# ============================================================
# .env 환경변수 로드
# ============================================================

load_dotenv()

# ===== MySQL DB =====
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "capstone_db"),
    "charset": "utf8mb4",
}

# ===== Telegram =====
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_SEND_PHOTO = True

# ===== Gmail SMTP =====
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
EMAIL_SENDER = os.getenv("EMAIL_SENDER", "")
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD", "")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", "")

# ===== 기능 토글 =====
ENABLE_DB_LOGGING = os.getenv("ENABLE_DB_SAVE", "True") == "True"
ENABLE_TELEGRAM_ALERT = os.getenv("ENABLE_TELEGRAM_ALERT", "True") == "True"
ENABLE_EMAIL_ALERT = os.getenv("ENABLE_EMAIL_ALERT", "True") == "True"


print("=" * 60)
print("[환경변수 로드 확인]")
print(f"  DB: {DB_CONFIG['user']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
print(f"  Telegram Token: {'OK' if TELEGRAM_BOT_TOKEN else 'NONE'}")
print(f"  Telegram Chat ID: {'OK' if TELEGRAM_CHAT_ID else 'NONE'}")
print(f"  Email Sender: {'OK' if EMAIL_SENDER else 'NONE'}")
print(f"  Email App Password: {'OK' if EMAIL_APP_PASSWORD else 'NONE'}")
print(f"  Toggle: DB={ENABLE_DB_LOGGING} | Telegram={ENABLE_TELEGRAM_ALERT} | Email={ENABLE_EMAIL_ALERT}")
print("=" * 60)


# ============================================================
# 얼굴 인식 설정
# ============================================================

# 2단계 판별 기준
REGISTER_THRESHOLD = 0.30
UNKNOWN_THRESHOLD = 0.47

# 실시간 안정화 전까지 Early Exit 비활성화
EARLY_EXIT_DISTANCE = -1

# opencv detector는 오탐 가능성이 있어 일단 retinaface만 사용
DETECTOR_BACKENDS = ["retinaface"]

MODEL_NAME = "ArcFace"

# YOLO person box 여유 영역
BOX_MARGIN_RATIO = 0.12

# 얼굴 품질 기준
MIN_FACE_WIDTH = 45
MIN_FACE_HEIGHT = 45
MIN_FACE_AREA_RATIO = 0.01
FACE_BORDER_MARGIN = 5
MIN_EYE_DISTANCE = 5

# YOLO 사람 탐지 신뢰도 기준
YOLO_CONF_THRESHOLD = 0.5


# ============================================================
# 경로 설정
# ============================================================

REGISTERED_DIR = Path("registered_faces/person01")
UPLOAD_DIR = Path("uploads")
ANNOTATED_DIR = Path("results/annotated")
ALERT_DIR = Path("results/alerts")
DEBUG_CROP_DIR = Path("results/debug_crops")

for d in [UPLOAD_DIR, ANNOTATED_DIR, ALERT_DIR, DEBUG_CROP_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# 알림 쿨다운
# ============================================================

ALERT_COOLDOWN_SEC = 0
last_alert_time = {}
cooldown_lock = threading.Lock()


# ============================================================
# YOLO 모델 로드
# ============================================================

print("[YOLO] 모델 로딩...")
yolo_model = YOLO("yolov8n.pt")
print("[YOLO] 로드 완료")


# ============================================================
# 유틸 함수
# ============================================================

def cosine_distance(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
    return 1.0 - np.dot(a, b) / denom


def is_valid_face(face_obj, crop_shape):
    """
    얼굴 품질 검증.
    너무 작거나, crop 경계에 붙어 있거나, 눈 거리 정보가 너무 작은 경우 제외.
    """
    try:
        fa = face_obj.get("facial_area", {})

        x = int(fa.get("x", 0))
        y = int(fa.get("y", 0))
        w = int(fa.get("w", 0))
        h = int(fa.get("h", 0))

        img_h, img_w = crop_shape[:2]

        if w < MIN_FACE_WIDTH or h < MIN_FACE_HEIGHT:
            return False

        face_area_ratio = (w * h) / float(img_w * img_h + 1e-8)
        if face_area_ratio < MIN_FACE_AREA_RATIO:
            return False

        if x <= FACE_BORDER_MARGIN or y <= FACE_BORDER_MARGIN:
            return False

        if x + w >= img_w - FACE_BORDER_MARGIN:
            return False

        if y + h >= img_h - FACE_BORDER_MARGIN:
            return False

        left_eye = fa.get("left_eye")
        right_eye = fa.get("right_eye")

        if left_eye and right_eye:
            eye_dist = ((left_eye[0] - right_eye[0]) ** 2 + (left_eye[1] - right_eye[1]) ** 2) ** 0.5
            if eye_dist < MIN_EYE_DISTANCE:
                return False

        return True

    except Exception:
        return False


def make_crops(img, x1, y1, x2, y2):
    """
    YOLO가 잡은 person box 기준으로 full / upper crop 생성.
    """
    h, w = img.shape[:2]

    bw = x2 - x1
    bh = y2 - y1

    if bw <= 0 or bh <= 0:
        return {}

    mx = int(bw * BOX_MARGIN_RATIO)
    my = int(bh * BOX_MARGIN_RATIO)

    ex1 = max(0, x1 - mx)
    ey1 = max(0, y1 - my)
    ex2 = min(w, x2 + mx)
    ey2 = min(h, y2 + my)

    full = img[ey1:ey2, ex1:ex2]

    crops = {}

    if full is not None and full.size > 0:
        crops["full"] = full

        upper_h = int(full.shape[0] * 0.45)
        if upper_h > 0:
            upper = full[:upper_h, :]
            if upper is not None and upper.size > 0:
                crops["upper"] = upper

    return crops


def save_debug_crop(crop_img, img_name, person_idx, crop_type):
    """
    디버깅용 crop 저장.
    어떤 crop으로 얼굴 비교했는지 확인 가능.
    """
    try:
        crop_path = DEBUG_CROP_DIR / f"{Path(img_name).stem}_p{person_idx}_{crop_type}.jpg"
        cv2.imwrite(str(crop_path), crop_img)
        return str(crop_path)
    except Exception:
        return None


# ============================================================
# 등록 얼굴 임베딩 사전 캐싱
# ============================================================

def precompute_registered_embeddings():
    cache = {d: [] for d in DETECTOR_BACKENDS}

    if not REGISTERED_DIR.exists():
        print(f"[경고] 등록 폴더 없음: {REGISTERED_DIR}")
        return cache

    images = [
        p for p in REGISTERED_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    print(f"[등록] 얼굴 {len(images)}장 임베딩 중...")
    print(f"[등록 폴더] {REGISTERED_DIR.resolve()}")

    for p in images:
        img = cv2.imread(str(p))
        if img is None:
            print(f"[등록 실패] {p.name}: 이미지 로드 실패")
            continue

        for detector in DETECTOR_BACKENDS:
            try:
                rep_list = DeepFace.represent(
                    img_path=str(p),
                    model_name=MODEL_NAME,
                    detector_backend=detector,
                    enforce_detection=True,
                    align=True,
                )

                if not rep_list:
                    print(f"[등록 실패] {p.name} / {detector}: 얼굴 없음")
                    continue

                best_face = None
                best_area = -1

                for face in rep_list:
                    if not is_valid_face(face, img.shape):
                        continue

                    fa = face.get("facial_area", {})
                    area = int(fa.get("w", 0)) * int(fa.get("h", 0))

                    if area > best_area:
                        best_area = area
                        best_face = face

                if best_face is None:
                    print(f"[등록 제외] {p.name} / {detector}: 품질 기준 미달")
                    continue

                cache[detector].append({
                    "name": p.name,
                    "embedding": np.asarray(best_face["embedding"], dtype=np.float32),
                })

                print(f"[등록 완료] {p.name} / {detector}")

            except Exception as e:
                print(f"[등록 실패] {p.name} / {detector}: {e}")

    for detector in DETECTOR_BACKENDS:
        print(f"   {detector}: {len(cache[detector])}개 캐시 완료")

    return cache


EMBEDDINGS_CACHE = precompute_registered_embeddings()

total_registered = sum(len(v) for v in EMBEDDINGS_CACHE.values())
if total_registered == 0:
    print("[경고] 등록 얼굴 임베딩이 0개입니다. registered_faces/person01 폴더를 확인하세요.")


# ============================================================
# 얼굴 비교 함수
# ============================================================

def compare_face(crops, img_name="unknown", person_idx=1):
    """
    crop 이미지에서 얼굴을 검출하고, 등록 얼굴 임베딩과 비교.

    판별 기준:
    distance < REGISTER_THRESHOLD       -> Registered
    REGISTER_THRESHOLD 이상 UNKNOWN_THRESHOLD 미만    -> Unidentified
    distance >= UNKNOWN_THRESHOLD       -> Unknown
    """
    best = {
        "final_result": "Unidentified",
        "distance": None,
        "detector": None,
        "crop_type": None,
        "matched": None,
    }

    if total_registered == 0:
        best["final_result"] = "Unidentified"
        return best

    for crop_type, crop_img in crops.items():
        if crop_img is None or crop_img.size == 0:
            continue

        save_debug_crop(crop_img, img_name, person_idx, crop_type)

        for detector in DETECTOR_BACKENDS:
            try:
                rep_list = DeepFace.represent(
                    img_path=crop_img,
                    model_name=MODEL_NAME,
                    detector_backend=detector,
                    enforce_detection=True,
                    align=True,
                )

                if not rep_list:
                    continue

                for face in rep_list:
                    if not is_valid_face(face, crop_img.shape):
                        continue

                    input_emb = np.asarray(face["embedding"], dtype=np.float32)
                    reg_list = EMBEDDINGS_CACHE.get(detector, [])

                    for reg in reg_list:
                        dist = cosine_distance(input_emb, reg["embedding"])

                        if best["distance"] is None or dist < best["distance"]:
                            best.update({
                                "distance": float(dist),
                                "detector": detector,
                                "crop_type": crop_type,
                                "matched": reg["name"],
                            })

                    if (
                        EARLY_EXIT_DISTANCE >= 0
                        and best["distance"] is not None
                        and best["distance"] < EARLY_EXIT_DISTANCE
                    ):
                        best["final_result"] = "Registered"
                        return best

            except Exception:
                continue

    if best["distance"] is None:
        best["final_result"] = "Unidentified"

    elif best["distance"] < REGISTER_THRESHOLD:
        best["final_result"] = "Registered"

    elif best["distance"] >= UNKNOWN_THRESHOLD:
        best["final_result"] = "Unknown"

    else:
        best["final_result"] = "Unidentified"

    return best


# ============================================================
# 이벤트 매핑
# ============================================================

def get_event_info(final_result):
    mapping = {
        "Registered": (
            "NORMAL_ACCESS",
            False,
            "Registered person detected."
        ),
        "Unknown": (
            "UNKNOWN_PERSON_DETECTED",
            True,
            "Alert: Unknown person detected."
        ),
        "Unidentified": (
            "FACE_NOT_IDENTIFIED",
            True,
            "Alert: Face could not be identified or result is ambiguous."
        ),
        "NoPerson": (
            "NO_PERSON",
            False,
            "No person detected."
        ),
        "CropFailed": (
            "CROP_FAILED",
            True,
            "Alert: Crop failed."
        ),
    }

    event_type, alert_required, alert_message = mapping.get(
        final_result,
        ("UNKNOWN_EVENT", True, "Unknown event")
    )

    return {
        "event_type": event_type,
        "alert_required": alert_required,
        "alert_message": alert_message,
    }


# ============================================================
# 쿨다운 체크
# ============================================================

def is_in_cooldown(event_type):
    with cooldown_lock:
        now = time.time()
        last = last_alert_time.get(event_type, 0)

        if now - last < ALERT_COOLDOWN_SEC:
            return True

        last_alert_time[event_type] = now
        return False


# ============================================================
# Telegram 알림
# ============================================================

def _send_telegram(event_info, row, image_path=None):
    caption = (
        f"[{event_info['event_type']}]\n"
        f"{event_info['alert_message']}\n\n"
        f"이미지: {row['image_name']}\n"
        f"결과: {row['final_result']}\n"
        f"distance: {row['distance']}\n"
        f"matched: {row['matched']}\n"
        f"detector: {row['detector']}\n"
        f"crop: {row['crop_type']}\n"
        f"시간: {row['detected_at']}"
    )

    try:
        if TELEGRAM_SEND_PHOTO and image_path and Path(image_path).exists():
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            with open(image_path, "rb") as f:
                files = {"photo": f}
                data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
                r = requests.post(url, data=data, files=files, timeout=10)
        else:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            data = {"chat_id": TELEGRAM_CHAT_ID, "text": caption}
            r = requests.post(url, data=data, timeout=10)

        if r.status_code == 200:
            print(f"[Telegram] 전송 완료: {event_info['event_type']}")
        else:
            print(f"[Telegram] 실패: {r.status_code} {r.text}")

    except Exception as e:
        print(f"[Telegram] 예외: {e}")


# ============================================================
# Email 알림
# ============================================================

def _send_email(event_info, row, image_path=None):
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER
        msg["Subject"] = f"[Capstone Alert] {event_info['event_type']}"

        body = f"""
침입 감지 시스템 알림

[이벤트 정보]
이벤트 종류    : {event_info['event_type']}
알림 메시지    : {event_info['alert_message']}
최종 판별 결과 : {row['final_result']}

[탐지 정보]
이미지명       : {row['image_name']}
YOLO 신뢰도    : {row['confidence']}
DeepFace 거리  : {row['distance']}
Detector       : {row['detector']}
Crop Type      : {row['crop_type']}
매칭 이미지     : {row['matched']}
탐지 시간      : {row['detected_at']}
""".strip()

        msg.attach(MIMEText(body, "plain", "utf-8"))

        if image_path and Path(image_path).exists():
            with open(image_path, "rb") as f:
                img = MIMEImage(f.read())
                img.add_header(
                    "Content-Disposition",
                    "attachment",
                    filename=Path(image_path).name
                )
                msg.attach(img)

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=20)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_APP_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        server.quit()

        print(f"[Email] 전송 완료: {event_info['event_type']}")

    except Exception as e:
        print(f"[Email] 예외: {e}")


# ============================================================
# 통합 알림
# ============================================================

def send_alerts_async(event_info, row, image_path=None):
    if not event_info["alert_required"]:
        return

    if is_in_cooldown(event_info["event_type"]):
        print(f"[쿨다운] {event_info['event_type']} 알림 생략")
        return

    if ENABLE_TELEGRAM_ALERT and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        threading.Thread(
            target=_send_telegram,
            args=(event_info, row, image_path),
            daemon=True
        ).start()

    if ENABLE_EMAIL_ALERT and EMAIL_SENDER and EMAIL_APP_PASSWORD and EMAIL_RECEIVER:
        threading.Thread(
            target=_send_email,
            args=(event_info, row, image_path),
            daemon=True
        ).start()


# ============================================================
# DB 저장
# ============================================================

def save_to_db(row):
    if not ENABLE_DB_LOGGING:
        return

    try:
        conn = pymysql.connect(**DB_CONFIG)
        cur = conn.cursor()

        sql = """
        INSERT INTO detection_log
        (test_group, model_name, image_name, person_idx, confidence,
         final_result, best_distance, best_detector, best_crop_type,
         event_type, alert_required, alert_message, annotated_path, detected_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        cur.execute(sql, (
            row["test_group"],
            row["model_name"],
            row["image_name"],
            row["person_idx"],
            row["confidence"],
            row["final_result"],
            row["distance"],
            row["detector"],
            row["crop_type"],
            row["event_type"],
            row["alert_required"],
            row["alert_message"],
            row["annotated_path"],
            row["detected_at"],
        ))

        conn.commit()
        cur.close()
        conn.close()

    except Exception as e:
        print(f"[DB] 저장 실패: {e}")


def save_alert_log(event_info, row):
    if not event_info["alert_required"]:
        return

    log_file = ALERT_DIR / "alert_log.txt"

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(
            f"[{row['detected_at']}] "
            f"{event_info['event_type']} | "
            f"{row['image_name']} | "
            f"result={row['final_result']} | "
            f"dist={row['distance']} | "
            f"matched={row['matched']} | "
            f"{event_info['alert_message']}\n"
        )


# ============================================================
# FastAPI 앱
# ============================================================

app = FastAPI(title="실시간 침입 감지 서버")


@app.get("/")
def root():
    return {
        "status": "running",
        "service": "intrusion-detection",
        "version": "week8-two-threshold",
        "register_threshold": REGISTER_THRESHOLD,
        "unknown_threshold": UNKNOWN_THRESHOLD,
        "early_exit": EARLY_EXIT_DISTANCE,
        "detectors": DETECTOR_BACKENDS,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "registered_faces": {d: len(EMBEDDINGS_CACHE[d]) for d in DETECTOR_BACKENDS},
        "total_registered": total_registered,
        "register_threshold": REGISTER_THRESHOLD,
        "unknown_threshold": UNKNOWN_THRESHOLD,
        "early_exit_distance": EARLY_EXIT_DISTANCE,
        "detectors": DETECTOR_BACKENDS,
        "telegram_enabled": ENABLE_TELEGRAM_ALERT,
        "email_enabled": ENABLE_EMAIL_ALERT,
        "db_enabled": ENABLE_DB_LOGGING,
        "cooldown_sec": ALERT_COOLDOWN_SEC,
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    start = time.time()

    # 1. 이미지 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    img_name = f"{timestamp}_{file.filename}"
    img_path = UPLOAD_DIR / img_name

    contents = await file.read()

    with open(img_path, "wb") as f:
        f.write(contents)

    # 2. 이미지 로드
    image = cv2.imread(str(img_path))

    if image is None:
        return JSONResponse(
            status_code=400,
            content={"error": "이미지 로드 실패"}
        )

    # 3. YOLO person 탐지
    yolo_start = time.time()

    results = yolo_model(
        image,
        classes=[0],
        verbose=False
    )

    yolo_time = (time.time() - yolo_start) * 1000

    detections = []
    annotated = image.copy()

    # 사람 미탐지
    if not results or len(results[0].boxes) == 0:
        row = {
            "test_group": "realtime",
            "model_name": "yolov8n",
            "image_name": img_name,
            "person_idx": 0,
            "confidence": 0.0,
            "final_result": "NoPerson",
            "distance": None,
            "detector": None,
            "crop_type": None,
            "matched": None,
            "annotated_path": "",
            "detected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        event_info = get_event_info("NoPerson")
        row.update(event_info)
        save_to_db(row)

        return {
            "status": "ok",
            "result": "NoPerson",
            "yolo_time_ms": round(yolo_time, 2),
            "total_time_ms": round((time.time() - start) * 1000, 2),
            "detections": [],
        }

    # 4. 탐지된 사람마다 얼굴 인식
    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        conf = float(box.conf[0])

        if conf < YOLO_CONF_THRESHOLD:
            continue

        h, w = image.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        crops = make_crops(image, x1, y1, x2, y2)

        if not crops:
            match = {
                "final_result": "CropFailed",
                "distance": None,
                "detector": None,
                "crop_type": None,
                "matched": None,
            }
        else:
            match = compare_face(
                crops,
                img_name=img_name,
                person_idx=i + 1
            )

        event_info = get_event_info(match["final_result"])

        if match["final_result"] == "Registered":
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        label = f"{match['final_result']} ({conf:.2f})"

        if match["distance"] is not None:
            label += f" d={match['distance']:.3f}"

        if match["matched"] is not None:
            label += f" {match['matched']}"

        cv2.putText(
            annotated,
            label,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        annotated_path = ANNOTATED_DIR / f"annot_{img_name}"
        cv2.imwrite(str(annotated_path), annotated)

        row = {
            "test_group": "realtime",
            "model_name": "yolov8n",
            "image_name": img_name,
            "person_idx": i + 1,
            "confidence": conf,
            "final_result": match["final_result"],
            "distance": float(match["distance"]) if match["distance"] is not None else None,
            "detector": match["detector"],
            "crop_type": match["crop_type"],
            "matched": match["matched"],
            "annotated_path": str(annotated_path),
            "detected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        row.update(event_info)

        save_to_db(row)
        save_alert_log(event_info, row)
        send_alerts_async(event_info, row, image_path=str(annotated_path))

        detections.append({
            "person_idx": i + 1,
            "confidence": round(conf, 3),
            "final_result": match["final_result"],
            "distance": row["distance"],
            "detector": match["detector"],
            "crop_type": match["crop_type"],
            "matched": match["matched"],
            "event_type": event_info["event_type"],
            "alert_required": event_info["alert_required"],
        })

        print(
            f"[결과] person={i+1} "
            f"result={match['final_result']} "
            f"conf={conf:.3f} "
            f"dist={row['distance']} "
            f"matched={match['matched']} "
            f"detector={match['detector']} "
            f"crop={match['crop_type']}"
        )

    total_time = (time.time() - start) * 1000

    print(f"[처리 완료] {img_name} ({total_time:.1f}ms, {len(detections)}명)")

    return {
        "status": "ok",
        "image": img_name,
        "yolo_time_ms": round(yolo_time, 2),
        "total_time_ms": round(total_time, 2),
        "register_threshold": REGISTER_THRESHOLD,
        "unknown_threshold": UNKNOWN_THRESHOLD,
        "early_exit_distance": EARLY_EXIT_DISTANCE,
        "detections": detections,
    }


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("[서버 시작] 실시간 침입 감지 서버 - 2단계 threshold 버전")
    print(f"  등록 얼굴 수: {total_registered}개")
    print(f"  Registered 기준: distance < {REGISTER_THRESHOLD}")
    print(f"  Unknown 기준: distance >= {UNKNOWN_THRESHOLD}")
    print(f"  애매한 구간: {REGISTER_THRESHOLD} <= distance < {UNKNOWN_THRESHOLD} -> Unidentified")
    print(f"  Early Exit: {EARLY_EXIT_DISTANCE}")
    print(f"  Detectors: {DETECTOR_BACKENDS}")
    print(f"  Telegram: {'ON' if ENABLE_TELEGRAM_ALERT else 'OFF'}")
    print(f"  Email:    {'ON' if ENABLE_EMAIL_ALERT else 'OFF'}")
    print(f"  MySQL:    {'ON' if ENABLE_DB_LOGGING else 'OFF'}")
    print(f"  쿨다운:   {ALERT_COOLDOWN_SEC}초")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000)