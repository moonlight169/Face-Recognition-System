# main.py

import cv2
import numpy as np
import os
import time
from datetime import datetime
import config

# ==========================================
# ฟังก์ชันตรวจสอบการเช็คอินซ้ำ
# ==========================================
def already_checked_in_today(filename: str, student_name: str) -> bool:
    if not os.path.exists(filename):
        return False
    thai_name = config.THAI_NAMES.get(student_name, student_name)
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if f"| ชื่อ: {thai_name} |" in line:
                return True
    return False

# ==========================================
# ระบบเทรนและโหลดสมองกล
# ==========================================
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(config.CASCADE_PATH)

if os.path.exists(config.TRAINER_PATH):
    print(f"[ระบบ] พบไฟล์ '{config.TRAINER_PATH}' โหลดข้อมูลเดิมมาใช้งานทันที...")
    recognizer.read(config.TRAINER_PATH)
else:
    print(f"[ระบบ] ยังไม่มีไฟล์ '{config.TRAINER_PATH}' กำลังเทรนโมเดล...")
    if not os.path.exists(config.DATASET_PATH):
        os.makedirs(config.DATASET_PATH)
        print(f"[แจ้งเตือน] กรุณาใส่รูปในโฟลเดอร์ '{config.DATASET_PATH}' ก่อนรันใหม่")
        exit()

    face_samples, ids = [], []
    for folder_name in os.listdir(config.DATASET_PATH):
        folder_path = os.path.join(config.DATASET_PATH, folder_name)
        if os.path.isdir(folder_path) and folder_name.lower().startswith('id'):
            user_id = int(folder_name[2:])
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img = cv2.imread(os.path.join(folder_path, filename))
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = detector.detectMultiScale(gray, 1.2, 5)
                    for (x, y, w, h) in faces:
                        face_samples.append(gray[y:y+h, x:x+w])
                        ids.append(user_id)

    if face_samples:
        recognizer.train(face_samples, np.array(ids))
        recognizer.write(config.TRAINER_PATH)
        print("[ระบบ] เทรนสำเร็จ!")
    else:
        print("[ข้อผิดพลาด] ไม่พบข้อมูลใบหน้าในโฟลเดอร์ dataset")
        exit()

# ==========================================
# ระบบเปิดกล้องและเช็คชื่อ
# ==========================================
cap = cv2.VideoCapture(0)
last_id, detection_start_time, already_logged, check_in_status = -1, 0, False, ""

print("[ระบบ] กำลังเปิดกล้อง... (กด 'q' เพื่อปิด)")

while True:
    ret, img = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.2, 5)
    current_frame_id = -1

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        if confidence < 100:
            name = config.NAMES[id] if id < len(config.NAMES) else f"ID {id}"
            match_percent = f"{round(100 - (confidence / 3))}%"
            current_frame_id = id
        else:
            name = "Unknown"
            match_percent = ""

        # --- แสดงหลอดโหลด ---
        if confidence < 100 and id == last_id:
            elapsed = time.time() - detection_start_time
            if elapsed > 5.0:
                elapsed = 5.0
            bar_width = int((elapsed / 5.0) * w)
            cv2.rectangle(img, (x, y + h + 15), (x + w, y + h + 25), (80, 80, 80), cv2.FILLED)
            cv2.rectangle(img, (x, y + h + 15), (x + bar_width, y + h + 25), (0, 165, 255), cv2.FILLED)

            if already_logged:
                if check_in_status == "SUCCESS":
                    color, msg = (0, 255, 0), "SUCCESS LOGGED"
                elif check_in_status == "DUPLICATE":
                    color, msg = (0, 0, 255), "ALREADY CHECKED TODAY"
                else:
                    color, msg = (0, 0, 255), "ERROR"
                cv2.putText(img, msg, (x, y + h + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                cv2.putText(img, f"Holding... {round(5.0 - elapsed, 1)}s", (x, y + h + 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        cv2.rectangle(img, (x, y-40), (x+w, y), (0, 0, 0), cv2.FILLED)
        cv2.putText(img, f"{name} {match_percent}", (x+5, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # --- ลอจิกการบันทึก ---
    if current_frame_id != -1 and current_frame_id == last_id and not already_logged:
        if time.time() - detection_start_time >= 5.0:
            student_name = config.NAMES[current_frame_id]
            data = config.STUDENT_DATA.get(student_name, {'id': 'ไม่พบ', 'year': '-'})
            thai_name = config.THAI_NAMES.get(student_name, student_name)

            filename = f"attendance_{datetime.now().strftime('%Y-%m-%d')}.txt"

            if already_checked_in_today(filename, student_name):
                check_in_status = "DUPLICATE"
            else:
                with open(filename, 'a', encoding='utf-8') as f:
                    f.write(f"เวลา: {datetime.now().strftime('%H:%M:%S')} | ชื่อ: {thai_name} | ชั้นปี: {data['year']} | รหัส: {data['id']}\n")
                check_in_status = "SUCCESS"
            already_logged = True

    elif current_frame_id != last_id:
        last_id = current_frame_id
        detection_start_time = time.time()
        already_logged = False
        check_in_status = ""

    cv2.imshow('Face Recognition System', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()