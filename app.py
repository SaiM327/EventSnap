from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import os, uuid
from fastapi import HTTPException
import face_recognition
import json
import numpy as np




app = FastAPI()

# If your frontend is opened from a different origin/port, enable CORS.
# If you serve frontend from the same FastAPI server later, you can tighten this.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")

EVENT_DIR = os.path.join(STORAGE_DIR, "event1")
os.makedirs(EVENT_DIR, exist_ok=True)

print("BASE_DIR:", BASE_DIR)
print("EVENT_DIR:", EVENT_DIR)

PROFILE_DIR = os.path.join(STORAGE_DIR, "profile")
os.makedirs(PROFILE_DIR, exist_ok=True)

print("PROFILE_DIR:", PROFILE_DIR)



ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

# Serve files in backend/storage at /static
app.mount("/static", StaticFiles(directory=STORAGE_DIR), name="static")


def is_allowed(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXT


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/event1/upload")
async def upload_event1(photos: list[UploadFile] = File(...)):
    saved_urls = []

    for photo in photos:
        if not photo.filename or not is_allowed(photo.filename):
            continue

        _, ext = os.path.splitext(photo.filename.lower())
        new_name = f"{uuid.uuid4().hex}{ext}"
        save_path = os.path.join(EVENT_DIR, new_name)

        # Save photo
        content = await photo.read()
        with open(save_path, "wb") as f:
            f.write(content)

        # --- FACE ENCODING ---
        image = face_recognition.load_image_file(save_path)
        encodings = face_recognition.face_encodings(image)
        encoding_list = [e.tolist() for e in encodings]  # could be empty if no faces
        # Save JSON file next to photo
        with open(save_path + ".json", "w") as f:
            json.dump(encoding_list, f)

        saved_urls.append(f"/static/event1/{new_name}")

    return {"saved": saved_urls}

@app.get("/event1/photos")
def list_event1_photos():
    files = []
    for name in os.listdir(EVENT_DIR):
        if is_allowed(name):
            files.append(f"/static/event1/{name}")

    # Optional: newest-ish first (not perfect unless you sort by mtime)
    files.sort(reverse=True)
    return {"photos": files}

@app.delete("/event1/photos/{filename}")
def delete_event1_photo(filename: str):
    # Prevent path traversal
    safe_name = os.path.basename(filename)
    file_path = os.path.join(EVENT_DIR, safe_name)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    if not is_allowed(safe_name):
        raise HTTPException(status_code=400, detail="Invalid file type")

    os.remove(file_path)
    return {"deleted": safe_name}

PROFILE_FILENAME = "profile"  # we'll store exactly 1 current profile photo


@app.post("/profile/upload")
async def upload_profile(photo: UploadFile = File(...)):
    if not photo.filename or not is_allowed(photo.filename):
        raise HTTPException(status_code=400, detail="Invalid file type")

    _, ext = os.path.splitext(photo.filename.lower())
    new_name = f"{PROFILE_FILENAME}{ext}"
    save_path = os.path.join(PROFILE_DIR, new_name)

    # Delete old profile photo
    for existing in os.listdir(PROFILE_DIR):
        if existing.startswith(PROFILE_FILENAME + "."):
            try:
                os.remove(os.path.join(PROFILE_DIR, existing))
            except:
                pass

    content = await photo.read()
    with open(save_path, "wb") as f:
        f.write(content)

    # --- FACE ENCODING ---
    image = face_recognition.load_image_file(save_path)
    encodings = face_recognition.face_encodings(image)
    if encodings:
        encoding_list = encodings[0].tolist()  # convert numpy to list
        with open(os.path.join(PROFILE_DIR, "profile_encoding.json"), "w") as f:
            json.dump(encoding_list, f)
    else:
        print("No face found in profile photo")

    return {"photo_url": f"/static/profile/{new_name}"}



@app.get("/profile")
def get_profile():
    # return the current profile photo url if present
    for name in os.listdir(PROFILE_DIR):
        if name.startswith(PROFILE_FILENAME + ".") and is_allowed(name):
            return {"photo_url": f"/static/profile/{name}"}
    return {"photo_url": None}


@app.get("/event1/photos_with_me")
def photos_with_me():
    # Load profile encoding
    profile_path = os.path.join(PROFILE_DIR, "profile_encoding.json")
    if not os.path.exists(profile_path):
        return {"photos": []}  # no profile photo uploaded yet

    with open(profile_path, "r") as f:
        profile_encoding = np.array(json.load(f))

    matched_photos = []

    for name in os.listdir(EVENT_DIR):
        if not is_allowed(name):
            continue

        json_path = os.path.join(EVENT_DIR, name + ".json")
        if not os.path.exists(json_path):
            continue

        with open(json_path, "r") as f:
            photo_encodings = json.load(f)

        for e in photo_encodings:
            match = face_recognition.compare_faces([profile_encoding], np.array(e), tolerance=0.6)
            if match[0]:
                matched_photos.append(f"/static/event1/{name}")
                break  # one match is enough

    return {"photos": matched_photos}
