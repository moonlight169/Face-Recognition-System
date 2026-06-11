import cv2
import numpy as np
import os

# ==========================================
# ส่วนตั้งค่าเริ่มต้น (Configuration)
# ==========================================
dataset_path = 'dataset'
cascade_path = 'haarcascade_frontalface_default.xml'
trainer_path = 'trainer.yml'

names = ['None', 'Ittichai'] 

# ==========================================
# ส่วนที่ 1: ระบบเทรนและโหลดสมองกล (Training/Loading System)
# ==========================================
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cascade_path)

if os.path.exists(trainer_path):
    print(f"[ระบบ] พบไฟล์ '{trainer_path}' โหลดข้อมูลสมองกลเดิมมาใช้งานทันที...")
    recognizer.read(trainer_path)
else:
    print(f"[ระบบ] ยังไม่มีไฟล์ '{trainer_path}' กำลังอ่านข้อมูลจากโฟลเดอร์เพื่อเทรนโมเดล...")
    
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        print(f"[แจ้งเตือน] สร้างโฟลเดอร์ '{dataset_path}' ให้แล้ว! กรุณาสร้างโฟลเดอร์ Id1 ด้านในแล้วใส่รูปก่อนรันใหม่")
        exit()

    face_samples = []
    ids = []

    # กวาดหาโฟลเดอร์ย่อยทั้งหมดใน dataset
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        
        # เช็คว่าเป็นโฟลเดอร์ และชื่อขึ้นต้นด้วยคำว่า 'Id' หรือ 'id'
        if os.path.isdir(folder_path) and folder_name.lower().startswith('id'):
            # ตัดตัวอักษร 'Id' ออก เพื่อเอาแค่ตัวเลข (เช่น Id1 -> 1)
            try:
                user_id = int(folder_name[2:])
            except:
                print(f"[ข้ามโฟลเดอร์] ชื่อ {folder_name} ไม่ถูกต้อง (ต้องเป็น Id1, Id2 ...)")
                continue
            
            print(f"-> กำลังอ่านรูปภาพในโฟลเดอร์: {folder_name} (ID: {user_id})")
            
            # วิ่งเข้าไปอ่านไฟล์รูปภาพทั้งหมดที่อยู่ในโฟลเดอร์นี้
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                    image_path = os.path.join(folder_path, filename)
                    img = cv2.imread(image_path)
                    
                    if img is None: 
                        continue
                    
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # ตรวจจับใบหน้า
                    faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
                    for (x, y, w, h) in faces:
                        face_samples.append(gray[y:y+h, x:x+w])
                        ids.append(user_id)

    # สั่งเทรนและบันทึกไฟล์
    if len(face_samples) > 0:
        recognizer.train(face_samples, np.array(ids))
        recognizer.write(trainer_path)
        print(f"[ระบบ] เทรนสำเร็จ! เรียนรู้ใบหน้าไปทั้งหมด {len(face_samples)} รูป และสร้างไฟล์ '{trainer_path}' เรียบร้อย")
    else:
        print("[ข้อผิดพลาด] เทรนไม่สำเร็จ: หาใบหน้าคนในรูปไม่เจอ หรือไม่มีรูปในโฟลเดอร์เลยครับ")
        exit()

# ==========================================
# ส่วนที่ 2: ระบบเปิดกล้องจดจำใบหน้า (Recognition System)
# ==========================================
print("[ระบบ] กำลังเปิดกล้อง... (คลิกที่หน้าต่างวิดีโอแล้วกด 'q' เพื่อออก)")
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    if not ret:
        break
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        # วาดกรอบสี่เหลี่ยมสีเขียว
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # ส่งให้สมองกลทายว่าเป็นใคร และได้ค่าความห่าง (Dist) กลับมา
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        # ถ้าค่าความห่างน้อยกว่า 100 ถือว่าน่าจะใช่คนเดียวกัน
        if confidence < 100:
            name = names[id] if id < len(names) else f"ID {id}"
            
            # --- ปรับสูตรคำนวณเปอร์เซ็นต์ ---
            # หารด้วย 3 เพื่อปรับสเกลให้ตัวเลขเปอร์เซ็นต์ดูสมจริงและไม่ต่ำจนเกินไป
            percent = round(100 - (confidence / 3)) 
            
            # ดักไว้ไม่ให้เกิน 100% (เผื่อกรณีฟลุ๊คได้ค่าน้อยมากๆ)
            if percent > 100: percent = 100
            
            match_percent = f"{percent}%"
        else:
            name = "Unknown"
            match_percent = ""

        # ใส่พื้นหลังสีดำทึบใต้ข้อความ
        cv2.rectangle(img, (x, y-40), (x+w, y), (0, 0, 0), cv2.FILLED)
        
        # แสดงชื่อและเปอร์เซ็นต์
        cv2.putText(img, f"{name} {match_percent}", (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('Face Recognition System', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()