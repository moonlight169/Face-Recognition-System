import os

# ตั้งค่าโฟลเดอร์และ ID ที่ต้องการ
folder_path = 'id1'
user_id = 1  # ถ้าเป็นหน้าคุณ Light ใส่เลข 1, ถ้าหน้าเพื่อนคนที่สองค่อยแก้เป็นเลข 2

print("กำลังเริ่มเปลี่ยนชื่อไฟล์...")

# อ่านรายชื่อไฟล์ทั้งหมดในโฟลเดอร์
files = os.listdir(folder_path)
count = 1

for filename in files:
    # เช็คว่าเป็นไฟล์รูปภาพไหม (กันเหนียว)
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        # สร้างชื่อไฟล์เก่าและใหม่
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, f"{user_id}.{count}.jpg")
        
        # สั่งเปลี่ยนชื่อ
        try:
            os.rename(old_file, new_file)
            print(f"✅ เปลี่ยนชื่อ: {filename}  ->  {user_id}.{count}.jpg")
            count += 1
        except FileExistsError:
            print(f"⚠️ ข้ามไฟล์: มีชื่อ {user_id}.{count}.jpg อยู่แล้ว")
            count += 1

print(f"🎉 เสร็จเรียบร้อย! เปลี่ยนชื่อไปทั้งหมด {count - 1} ไฟล์")