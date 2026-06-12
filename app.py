import io
import json
import os
import shutil
import threading
import uuid
import zipfile

import face_recognition
import numpy as np
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from db import AuthToken, Event, Photo, SessionLocal, User, init_db
from security import (
    create_token,
    get_current_user,
    get_db,
    hash_password,
    verify_password,
)

app = FastAPI(title="EventSnap API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.environ.get("EVENTSNAP_STORAGE", os.path.join(BASE_DIR, "storage"))
os.makedirs(STORAGE_DIR, exist_ok=True)
EVENTS_DIR = os.path.join(STORAGE_DIR, "events")
PROFILES_DIR = os.path.join(STORAGE_DIR, "profiles")
os.makedirs(EVENTS_DIR, exist_ok=True)
os.makedirs(PROFILES_DIR, exist_ok=True)

ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10MB per file
FACE_MATCH_TOLERANCE = 0.6

app.mount("/static", StaticFiles(directory=STORAGE_DIR), name="static")


def is_allowed(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXT


def event_dir(event_id: int) -> str:
    path = os.path.join(EVENTS_DIR, str(event_id))
    os.makedirs(path, exist_ok=True)
    return path


def photo_url(photo: Photo) -> str:
    return f"/static/events/{photo.event_id}/{photo.filename}"


# ---------------------------------------------------------------------------
# Startup: create tables + one-time migration of legacy storage/event1 photos
# ---------------------------------------------------------------------------

def migrate_legacy_event1():
    legacy_dir = os.path.join(STORAGE_DIR, "event1")
    if not os.path.isdir(legacy_dir):
        return

    db = SessionLocal()
    try:
        if db.scalar(select(func.count()).select_from(Event)) > 0:
            return

        legacy_photos = [n for n in os.listdir(legacy_dir) if is_allowed(n)]
        if not legacy_photos:
            return

        event = Event(name="Event 1", event_date=None, created_by=None)
        db.add(event)
        db.flush()

        target_dir = event_dir(event.id)
        for name in legacy_photos:
            src = os.path.join(legacy_dir, name)
            shutil.move(src, os.path.join(target_dir, name))

            encodings = "[]"
            json_path = src + ".json"
            if os.path.exists(json_path):
                with open(json_path) as f:
                    encodings = f.read()
                os.remove(json_path)

            db.add(
                Photo(
                    event_id=event.id,
                    filename=name,
                    uploaded_by=None,
                    encodings=encodings,
                    encoded=True,
                )
            )

        db.commit()
        print(f"Migrated {len(legacy_photos)} legacy photos into event {event.id}")
    finally:
        db.close()


def resume_pending_encodings():
    """Re-encode photos whose background task never finished (e.g. server restart)."""
    db = SessionLocal()
    try:
        pending = db.scalars(select(Photo).where(Photo.encoded == False)).all()  # noqa: E712
        jobs = [
            (p.id, os.path.join(EVENTS_DIR, str(p.event_id), p.filename)) for p in pending
        ]
    finally:
        db.close()

    if not jobs:
        return

    def worker():
        for photo_id, path in jobs:
            encode_photo_task(photo_id, path)

    threading.Thread(target=worker, daemon=True).start()
    print(f"Resuming face encoding for {len(jobs)} photo(s)")


@app.on_event("startup")
def on_startup():
    init_db()
    migrate_legacy_event1()
    resume_pending_encodings()


# ---------------------------------------------------------------------------
# Face encoding (runs in background after upload)
# ---------------------------------------------------------------------------

# dlib is not thread-safe: concurrent encodings (e.g. a background task and a
# profile upload) can segfault the process. Serialize all encoding work.
ENCODING_LOCK = threading.Lock()


def compute_encodings(path: str) -> list:
    with ENCODING_LOCK:
        image = face_recognition.load_image_file(path)
        return face_recognition.face_encodings(image)


def encode_photo_task(photo_id: int, path: str):
    encodings_json = "[]"
    try:
        encodings = compute_encodings(path)
        encodings_json = json.dumps([e.tolist() for e in encodings])
    except Exception as exc:
        print(f"Face encoding failed for photo {photo_id}: {exc}")

    db = SessionLocal()
    try:
        photo = db.get(Photo, photo_id)
        if photo is not None:
            photo.encodings = encodings_json
            photo.encoded = True
            db.commit()
    finally:
        db.close()


def matched_photos_for_user(db: Session, event_id: int, user: User) -> tuple[list[Photo], int]:
    """Returns (matched photos, count of photos still awaiting encoding)."""
    profile = np.array(json.loads(user.profile_encoding))

    photos = db.scalars(
        select(Photo).where(Photo.event_id == event_id).order_by(Photo.created_at.desc())
    ).all()

    matched = []
    pending = 0
    for photo in photos:
        if not photo.encoded:
            pending += 1
            continue
        encodings = json.loads(photo.encodings or "[]")
        if not encodings:
            continue
        distances = np.linalg.norm(np.array(encodings) - profile, axis=1)
        if (distances <= FACE_MATCH_TOLERANCE).any():
            matched.append(photo)

    return matched, pending


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class ProfileUpdate(BaseModel):
    name: str
    email: EmailStr


class EventCreate(BaseModel):
    name: str
    event_date: str | None = None


def user_payload(user: User) -> dict:
    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "photo_url": f"/static/profiles/{user.profile_filename}" if user.profile_filename else None,
        "has_face_encoding": user.profile_encoding is not None,
    }


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

@app.post("/auth/register")
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    if len(body.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    existing = db.scalar(select(User).where(User.email == body.email))
    if existing:
        raise HTTPException(status_code=409, detail="An account with this email already exists")

    user = User(
        name=body.name.strip(),
        email=body.email,
        password_hash=hash_password(body.password),
    )
    db.add(user)
    db.commit()

    token = create_token(db, user)
    return {"token": token, "user": user_payload(user)}


@app.post("/auth/login")
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.email == body.email))
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_token(db, user)
    return {"token": token, "user": user_payload(user)}


@app.post("/auth/logout")
def logout(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    db.query(AuthToken).filter(AuthToken.user_id == user.id).delete()
    db.commit()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

@app.get("/me")
def get_me(user: User = Depends(get_current_user)):
    return user_payload(user)


@app.put("/me")
def update_me(
    body: ProfileUpdate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if body.email != user.email:
        taken = db.scalar(select(User).where(User.email == body.email))
        if taken:
            raise HTTPException(status_code=409, detail="Email already in use")

    user.name = body.name.strip()
    user.email = body.email
    db.commit()
    return user_payload(user)


@app.post("/me/photo")
def upload_profile_photo(
    photo: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not photo.filename or not is_allowed(photo.filename):
        raise HTTPException(status_code=400, detail="Invalid file type")

    content = photo.file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")

    _, ext = os.path.splitext(photo.filename.lower())
    new_name = f"{user.id}_{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(PROFILES_DIR, new_name)
    with open(save_path, "wb") as f:
        f.write(content)

    # Profile photos are encoded synchronously: matching depends on them
    # and it's a single image. (Sync endpoint -> runs in a worker thread.)
    face_found = False
    try:
        encodings = compute_encodings(save_path)
        if encodings:
            user.profile_encoding = json.dumps(encodings[0].tolist())
            face_found = True
    except Exception as exc:
        print(f"Profile encoding failed for user {user.id}: {exc}")

    # Remove the previous photo file
    if user.profile_filename:
        old = os.path.join(PROFILES_DIR, user.profile_filename)
        if os.path.exists(old):
            os.remove(old)

    user.profile_filename = new_name
    db.commit()

    return {
        "photo_url": f"/static/profiles/{new_name}",
        "face_found": face_found,
    }


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@app.get("/events")
def list_events(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rows = db.execute(
        select(Event, func.count(Photo.id))
        .outerjoin(Photo)
        .group_by(Event.id)
        .order_by(Event.created_at.desc())
    ).all()

    events = []
    for event, photo_count in rows:
        cover = db.scalar(
            select(Photo)
            .where(Photo.event_id == event.id)
            .order_by(Photo.created_at.desc())
            .limit(1)
        )
        events.append(
            {
                "id": event.id,
                "name": event.name,
                "event_date": event.event_date,
                "photo_count": photo_count,
                "cover_url": photo_url(cover) if cover else None,
                "is_owner": event.created_by in (None, user.id),
            }
        )
    return {"events": events}


@app.post("/events")
def create_event(
    body: EventCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Event name is required")

    event = Event(name=name, event_date=body.event_date, created_by=user.id)
    db.add(event)
    db.commit()
    return {"id": event.id, "name": event.name, "event_date": event.event_date}


def get_event_or_404(db: Session, event_id: int) -> Event:
    event = db.get(Event, event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return event


@app.get("/events/{event_id}")
def get_event(
    event_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    event = get_event_or_404(db, event_id)
    count = db.scalar(select(func.count()).select_from(Photo).where(Photo.event_id == event.id))
    return {
        "id": event.id,
        "name": event.name,
        "event_date": event.event_date,
        "photo_count": count,
        "is_owner": event.created_by in (None, user.id),
    }


@app.delete("/events/{event_id}")
def delete_event(
    event_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    event = get_event_or_404(db, event_id)
    if event.created_by is not None and event.created_by != user.id:
        raise HTTPException(status_code=403, detail="Only the event creator can delete it")

    db.delete(event)  # cascades to photo rows
    db.commit()

    shutil.rmtree(os.path.join(EVENTS_DIR, str(event_id)), ignore_errors=True)
    return {"deleted": event_id}


# ---------------------------------------------------------------------------
# Event photos
# ---------------------------------------------------------------------------

@app.post("/events/{event_id}/photos")
async def upload_event_photos(
    event_id: int,
    background_tasks: BackgroundTasks,
    photos: list[UploadFile] = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    event = get_event_or_404(db, event_id)
    target_dir = event_dir(event.id)

    saved = []
    for upload in photos:
        if not upload.filename or not is_allowed(upload.filename):
            continue

        content = await upload.read()
        if len(content) > MAX_UPLOAD_BYTES:
            continue

        _, ext = os.path.splitext(upload.filename.lower())
        new_name = f"{uuid.uuid4().hex}{ext}"
        save_path = os.path.join(target_dir, new_name)
        with open(save_path, "wb") as f:
            f.write(content)

        photo = Photo(
            event_id=event.id,
            filename=new_name,
            uploaded_by=user.id,
            encoded=False,
        )
        db.add(photo)
        db.flush()

        background_tasks.add_task(encode_photo_task, photo.id, save_path)
        saved.append(photo)

    db.commit()
    return {"saved": [photo_url(p) for p in saved], "encoding_in_background": len(saved)}


@app.get("/events/{event_id}/photos")
def list_event_photos(
    event_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    get_event_or_404(db, event_id)
    photos = db.scalars(
        select(Photo).where(Photo.event_id == event_id).order_by(Photo.created_at.desc())
    ).all()
    return {"photos": [{"id": p.id, "url": photo_url(p)} for p in photos]}


@app.delete("/events/{event_id}/photos/{photo_id}")
def delete_event_photo(
    event_id: int,
    photo_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    photo = db.get(Photo, photo_id)
    if photo is None or photo.event_id != event_id:
        raise HTTPException(status_code=404, detail="Photo not found")

    file_path = os.path.join(EVENTS_DIR, str(event_id), photo.filename)
    db.delete(photo)
    db.commit()

    if os.path.exists(file_path):
        os.remove(file_path)
    return {"deleted": photo_id}


@app.get("/events/{event_id}/photos_with_me")
def photos_with_me(
    event_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    get_event_or_404(db, event_id)

    if not user.profile_encoding:
        return {"photos": [], "pending": 0, "needs_profile": True}

    matched, pending = matched_photos_for_user(db, event_id, user)
    return {
        "photos": [{"id": p.id, "url": photo_url(p)} for p in matched],
        "pending": pending,
        "needs_profile": False,
    }


# ---------------------------------------------------------------------------
# Bulk download
# ---------------------------------------------------------------------------

@app.get("/events/{event_id}/download")
def download_event_photos(
    event_id: int,
    scope: str = "all",
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    event = get_event_or_404(db, event_id)

    if scope == "me":
        if not user.profile_encoding:
            raise HTTPException(status_code=400, detail="Upload a profile photo first")
        photos, _ = matched_photos_for_user(db, event_id, user)
    else:
        photos = db.scalars(
            select(Photo).where(Photo.event_id == event_id).order_by(Photo.created_at)
        ).all()

    if not photos:
        raise HTTPException(status_code=404, detail="No photos to download")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for photo in photos:
            path = os.path.join(EVENTS_DIR, str(event_id), photo.filename)
            if os.path.exists(path):
                zf.write(path, arcname=photo.filename)
    buffer.seek(0)

    safe_event_name = "".join(c if c.isalnum() else "_" for c in event.name) or "event"
    suffix = "my_photos" if scope == "me" else "all_photos"
    filename = f"{safe_event_name}_{suffix}.zip"

    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
