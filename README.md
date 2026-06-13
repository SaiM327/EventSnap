# EventSnap

EventSnap is a full-stack event photo sharing platform with facial recognition. Users create event albums, upload photos, and instantly discover every photo they appear in — the backend matches faces in event photos against each user's profile photo and builds a personalized "Photos with Me" gallery.

## Features

### Accounts & Authentication
- Register / login with token-based authentication
- Passwords hashed with PBKDF2 (200k iterations)
- Per-user profiles with name, email, and profile photo
- All API routes protected by bearer-token auth

### Event Albums
- Create unlimited event albums with names and dates
- Albums dashboard with cover photos and live photo counts
- Owner-only event deletion
- Shared albums: every user can browse events and find themselves in the photos

### Photo Uploads & Management
- Upload multiple photos at once
- File type and size validation, unique server-side filenames
- Delete individual photos
- Face encoding runs as a background task, so uploads return instantly
- Interrupted encoding jobs automatically resume on server restart

### Facial Recognition
- 128-dimension face embeddings generated for every detected face (dlib / face_recognition)
- Profile photo embedding stored per user
- "Photos with Me" tab compares embeddings with vectorized NumPy distance matching
- Reports how many photos are still being scanned

### Bulk Download
- Download an entire album as a zip
- Or download only the photos you appear in

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI, SQLAlchemy |
| Database | SQLite (users, sessions, events, photos, face embeddings) |
| Computer Vision | dlib via face_recognition, NumPy |
| Frontend | HTML5, CSS3, vanilla JavaScript |
| Testing | pytest (22 tests, isolated test DB, mocked encodings) |

## Project Structure

```
app.py              FastAPI application: auth, events, photos, matching, downloads
db.py               SQLAlchemy models: User, AuthToken, Event, Photo
security.py         Password hashing, token creation, auth dependency
cli.py              Admin CLI: stats, user management, cleanup, re-encoding
tests/test_api.py   End-to-end API test suite
index.html          Albums dashboard
event.html          Album page (upload, tabs, downloads)
profile.html        Profile settings
login.html          Sign in / create account
app.js              Shared frontend auth helpers
styles.css          Styling
```

## Getting Started

```bash
# 1. Create a virtual environment and install dependencies
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Start the API server
.venv/bin/uvicorn app:app --port 8001

# 3. Open login.html in a browser (or serve the folder with any static server)
```

The SQLite database (`eventsnap.db`) and storage folders are created automatically on first run.

### Run the Tests

```bash
.venv/bin/python -m pytest tests/
```

### Admin CLI

```bash
python cli.py stats              # user/event/photo counts
python cli.py list-users
python cli.py list-events
python cli.py reset-password EMAIL
python cli.py encode-pending     # encode photos that were never processed
python cli.py cleanup            # remove orphaned files and DB rows
```

## How It Works

1. **Sign up** and upload a profile photo — a 128-d face embedding is extracted and stored on your account.
2. **Create an event** and upload photos. Each photo is saved immediately; face detection runs in the background (serialized behind a lock, since dlib is not thread-safe).
3. **Open "Photos with Me"** — the backend compares your profile embedding against every face found in the album using NumPy vectorized Euclidean distance (tolerance 0.6) and returns only your photos.
4. **Download** the full album or just your matches as a zip.

## Future Improvements

- Cloud storage integration (AWS S3)
- FAISS-powered embedding search for large albums
- Thumbnails for faster gallery loading
- Share links so guests can join an event without an account
- Automatic face clustering and tagging
- Mobile application support
