# 설치 및 실행 안내

## 1. 실행 환경

서버는 Python, YOLOv8n, DeepFace, OpenCV를 사용합니다. 원래 실험 환경의 Python·CUDA·패키지 버전 잠금 파일은 제공되지 않아 특정 버전 조합을 검증된 환경으로 제시하지 않습니다. 별도의 가상환경에서 `requirements.txt`를 설치하고, TensorFlow/Keras 관련 오류가 있으면 실제 설치 버전의 호환성을 확인합니다.

이미 사용 중인 정상 동작 환경이 있다면 먼저 그 환경에서 실행하고 `python -m pip freeze`로 버전을 기록하는 것이 재현에 도움이 됩니다.

## 2. 경로와 등록 얼굴

상대 경로를 사용하므로 반드시 저장소 루트에서 실행합니다.

- `.env.example`을 `.env`로 복사합니다.
- `yolov8n.pt`를 루트에 배치합니다. `yolov8s.pt`는 최종 서버 실행에 사용하지 않습니다.
- `registered_faces/person01/`에 JPG, JPEG, PNG 사진을 배치합니다. 하위 폴더는 재귀 탐색하지 않습니다.
- 시작할 때 등록 얼굴 임베딩을 생성합니다. 사진을 변경했다면 서버를 재시작합니다.
- 얼굴 품질 검사를 통과하지 못하면 사진이 있어도 등록 임베딩이 생성되지 않을 수 있습니다. `/health`의 `total_registered`와 서버 로그를 확인합니다.

현재 코드는 하나의 `person01` 폴더를 사용하며, 매칭 결과는 등록 이미지 파일명입니다. 별도의 사용자 등록·관리 API는 없습니다.

## 3. 선택적 DB 연결

`sql/schema.sql`은 서버의 INSERT 컬럼을 기준으로 새로 작성한 예시입니다. 기존 DB의 원본 스키마 덤프가 아닙니다. 새 개발 DB에서 준비하거나 기존 테이블과 컬럼을 비교한 뒤 사용합니다. `CREATE TABLE IF NOT EXISTS`는 기존 테이블 구조를 변경하지 않습니다.

MySQL 클라이언트에서:

```sql
SOURCE sql/schema.sql;
```

접속 계정과 필요한 권한은 별도로 준비하고 `.env`에 설정합니다.

```dotenv
DB_HOST=127.0.0.1
DB_PORT=3306
DB_USER=capstone_user
DB_PASSWORD=your_local_database_password
DB_NAME=capstone_db
ENABLE_DB_SAVE=True
```

DB 저장이 실패해도 서버는 오류를 로그에 남기고 분석 응답을 계속 반환할 수 있습니다. HTTP 성공 응답만으로 저장 성공을 판단하지 말고 실제 테이블을 확인합니다.

## 4. Telegram·이메일

Telegram은 `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`를 설정하고 `ENABLE_TELEGRAM_ALERT=True`로 변경합니다.

이메일은 `SMTP_SERVER`, `SMTP_PORT`, `EMAIL_SENDER`, `EMAIL_APP_PASSWORD`, `EMAIL_RECEIVER`를 설정하고 `ENABLE_EMAIL_ALERT=True`로 변경합니다. 코드의 SMTP 연결은 STARTTLS 방식을 사용합니다.

토글은 코드에서 문자열 `True`와 정확히 비교합니다. `true`나 `1`을 사용하지 않습니다. `.env`를 수정한 뒤 서버를 재시작합니다. 알림 대상 이미지는 설정한 외부 채널로 전달되므로 수신 대상을 확인합니다.

## 5. 실행 및 API 확인

```bash
python server_realtime.py
```

서버는 `0.0.0.0:8000`에서 수신합니다. 같은 PC에서는 `http://127.0.0.1:8000/docs`에 접속하고, Raspberry Pi에서는 PC의 실험 네트워크 주소를 사용합니다. 서버 포트에 대한 네트워크·방화벽 설정을 확인합니다.

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/detect -F "file=@sample.jpg"
```

Windows PowerShell에서는 curl 별칭과 구분하기 위해 `curl.exe`를 사용할 수 있습니다. `sample.jpg`는 직접 준비한 입력 이미지입니다.

`/health`의 DB·알림 활성화 값은 설정 상태이며 실제 연결 검사가 아닙니다. 등록·미등록·얼굴 식별 불가·사람 없음 이미지를 각각 전송하고, 필요한 경우 DB 저장과 알림 수신까지 확인합니다.

## 6. 설정 상수

아래 항목은 `.env`가 아니라 `server_realtime.py`에서 수정합니다.

| 상수 | 현재 값 | 의미 |
|---|---|---|
| `REGISTER_THRESHOLD` | 0.30 | 등록 판별 경계 |
| `UNKNOWN_THRESHOLD` | 0.47 | 미등록 판별 경계 |
| `EARLY_EXIT_DISTANCE` | -1 | 조기 종료 비활성화 |
| `DETECTOR_BACKENDS` | retinaface | 얼굴 검출기 |
| `YOLO_CONF_THRESHOLD` | 0.5 | 탐지 후 신뢰도 필터 |
| `ALERT_COOLDOWN_SEC` | 0 | 알림 쿨다운 없음 |

30초간 같은 종류의 알림을 제한하려면 `ALERT_COOLDOWN_SEC = 30`으로 변경합니다. 쿨다운은 사람별이 아닌 이벤트 종류별이며 알림 발송 성공 여부와 독립적으로 갱신됩니다.
