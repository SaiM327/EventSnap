"""End-to-end API tests for EventSnap.

Face encoding is faked (see fake_compute_encodings) so the suite runs fast
and deterministically without dlib doing real face detection: the byte
content of an uploaded file decides which "face" it contains.
"""

import io
import os
import tempfile
import zipfile

# Point the app at a throwaway DB and storage dir BEFORE importing it.
_TMP = tempfile.mkdtemp(prefix="eventsnap_test_")
os.environ["EVENTSNAP_DB"] = os.path.join(_TMP, "test.db")
os.environ["EVENTSNAP_STORAGE"] = os.path.join(_TMP, "storage")

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as app_module
from app import app

# ---------------------------------------------------------------------------
# Fake faces: 128-dim vectors. ALICE and ALICE_VARIANT are within the match
# tolerance of each other; BOB is far away from both.
# ---------------------------------------------------------------------------

ALICE = np.zeros(128)
ALICE_VARIANT = np.full(128, 0.01)
BOB = np.ones(128)

FACE_BY_CONTENT = {
    b"photo-of-alice": [ALICE_VARIANT],
    b"photo-of-bob": [BOB],
    b"photo-of-both": [ALICE_VARIANT, BOB],
    b"alice-profile": [ALICE],
    b"no-face-here": [],
}


def fake_compute_encodings(path):
    with open(path, "rb") as f:
        content = f.read()
    return FACE_BY_CONTENT.get(content, [])


@pytest.fixture(autouse=True)
def patch_encodings(monkeypatch):
    monkeypatch.setattr(app_module, "compute_encodings", fake_compute_encodings)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def register(client, name="Alice", email="alice@example.com", password="secret123"):
    res = client.post(
        "/auth/register",
        json={"name": name, "email": email, "password": password},
    )
    assert res.status_code == 200, res.text
    data = res.json()
    return {"Authorization": f"Bearer {data['token']}"}, data["user"]


def upload_file(name: str, content: bytes):
    return ("photos", (name, io.BytesIO(content), "image/jpeg"))


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class TestAuth:
    def test_health(self, client):
        assert client.get("/health").json() == {"status": "ok"}

    def test_register_and_login(self, client):
        headers, user = register(client, email="auth-test@example.com")
        assert user["name"] == "Alice"

        res = client.post(
            "/auth/login",
            json={"email": "auth-test@example.com", "password": "secret123"},
        )
        assert res.status_code == 200
        assert res.json()["user"]["email"] == "auth-test@example.com"

    def test_duplicate_email_rejected(self, client):
        register(client, email="dupe@example.com")
        res = client.post(
            "/auth/register",
            json={"name": "X", "email": "dupe@example.com", "password": "secret123"},
        )
        assert res.status_code == 409

    def test_short_password_rejected(self, client):
        res = client.post(
            "/auth/register",
            json={"name": "X", "email": "short@example.com", "password": "abc"},
        )
        assert res.status_code == 400

    def test_wrong_password_rejected(self, client):
        register(client, email="wrongpw@example.com")
        res = client.post(
            "/auth/login",
            json={"email": "wrongpw@example.com", "password": "not-the-password"},
        )
        assert res.status_code == 401

    def test_protected_routes_require_token(self, client):
        assert client.get("/events").status_code == 401
        assert client.get("/me").status_code == 401

        bad = {"Authorization": "Bearer not-a-real-token"}
        assert client.get("/events", headers=bad).status_code == 401

    def test_logout_invalidates_token(self, client):
        headers, _ = register(client, email="logout@example.com")
        assert client.get("/me", headers=headers).status_code == 200
        assert client.post("/auth/logout", headers=headers).status_code == 200
        assert client.get("/me", headers=headers).status_code == 401


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

class TestProfile:
    def test_update_profile(self, client):
        headers, _ = register(client, email="update-me@example.com")
        res = client.put(
            "/me",
            headers=headers,
            json={"name": "New Name", "email": "renamed@example.com"},
        )
        assert res.status_code == 200
        assert res.json()["name"] == "New Name"
        assert res.json()["email"] == "renamed@example.com"

    def test_update_to_taken_email_rejected(self, client):
        register(client, email="taken@example.com")
        headers, _ = register(client, email="other@example.com")
        res = client.put(
            "/me",
            headers=headers,
            json={"name": "X", "email": "taken@example.com"},
        )
        assert res.status_code == 409

    def test_profile_photo_with_face(self, client):
        headers, _ = register(client, email="face@example.com")
        res = client.post(
            "/me/photo",
            headers=headers,
            files={"photo": ("me.jpg", io.BytesIO(b"alice-profile"), "image/jpeg")},
        )
        assert res.status_code == 200
        assert res.json()["face_found"] is True
        assert client.get("/me", headers=headers).json()["has_face_encoding"] is True

    def test_profile_photo_without_face(self, client):
        headers, _ = register(client, email="no-face@example.com")
        res = client.post(
            "/me/photo",
            headers=headers,
            files={"photo": ("me.jpg", io.BytesIO(b"no-face-here"), "image/jpeg")},
        )
        assert res.status_code == 200
        assert res.json()["face_found"] is False

    def test_profile_photo_bad_extension(self, client):
        headers, _ = register(client, email="bad-ext@example.com")
        res = client.post(
            "/me/photo",
            headers=headers,
            files={"photo": ("me.txt", io.BytesIO(b"alice-profile"), "text/plain")},
        )
        assert res.status_code == 400


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

class TestEvents:
    def test_create_and_list(self, client):
        headers, _ = register(client, email="events@example.com")
        res = client.post(
            "/events",
            headers=headers,
            json={"name": "Birthday Bash", "event_date": "2026-08-01"},
        )
        assert res.status_code == 200
        event_id = res.json()["id"]

        events = client.get("/events", headers=headers).json()["events"]
        match = next(e for e in events if e["id"] == event_id)
        assert match["name"] == "Birthday Bash"
        assert match["photo_count"] == 0
        assert match["is_owner"] is True

    def test_blank_name_rejected(self, client):
        headers, _ = register(client, email="blank@example.com")
        res = client.post("/events", headers=headers, json={"name": "   "})
        assert res.status_code == 400

    def test_get_missing_event(self, client):
        headers, _ = register(client, email="missing@example.com")
        assert client.get("/events/99999", headers=headers).status_code == 404

    def test_only_owner_can_delete(self, client):
        owner_headers, _ = register(client, email="owner@example.com")
        other_headers, _ = register(client, email="intruder@example.com")

        event_id = client.post(
            "/events", headers=owner_headers, json={"name": "Private"}
        ).json()["id"]

        assert client.delete(f"/events/{event_id}", headers=other_headers).status_code == 403
        assert client.delete(f"/events/{event_id}", headers=owner_headers).status_code == 200
        assert client.get(f"/events/{event_id}", headers=owner_headers).status_code == 404


# ---------------------------------------------------------------------------
# Photos + face matching + downloads
# ---------------------------------------------------------------------------

class TestPhotos:
    @pytest.fixture()
    def event(self, client):
        headers, _ = register(
            client, email=f"photos-{os.urandom(4).hex()}@example.com"
        )
        event_id = client.post(
            "/events", headers=headers, json={"name": "Party"}
        ).json()["id"]
        return client, headers, event_id

    def test_upload_list_delete(self, event):
        client, headers, event_id = event

        res = client.post(
            f"/events/{event_id}/photos",
            headers=headers,
            files=[
                upload_file("a.jpg", b"photo-of-alice"),
                upload_file("b.jpg", b"photo-of-bob"),
                upload_file("evil.exe", b"not-a-photo"),  # filtered out
            ],
        )
        assert res.status_code == 200
        assert len(res.json()["saved"]) == 2

        photos = client.get(f"/events/{event_id}/photos", headers=headers).json()["photos"]
        assert len(photos) == 2

        photo_id = photos[0]["id"]
        res = client.delete(f"/events/{event_id}/photos/{photo_id}", headers=headers)
        assert res.status_code == 200

        photos = client.get(f"/events/{event_id}/photos", headers=headers).json()["photos"]
        assert len(photos) == 1

    def test_delete_photo_from_wrong_event(self, event):
        client, headers, event_id = event
        other_event = client.post(
            "/events", headers=headers, json={"name": "Other"}
        ).json()["id"]

        client.post(
            f"/events/{event_id}/photos",
            headers=headers,
            files=[upload_file("a.jpg", b"photo-of-alice")],
        )
        photo_id = client.get(
            f"/events/{event_id}/photos", headers=headers
        ).json()["photos"][0]["id"]

        res = client.delete(f"/events/{other_event}/photos/{photo_id}", headers=headers)
        assert res.status_code == 404

    def test_photos_with_me_requires_profile(self, event):
        client, headers, event_id = event
        data = client.get(f"/events/{event_id}/photos_with_me", headers=headers).json()
        assert data["needs_profile"] is True
        assert data["photos"] == []

    def test_face_matching(self, event):
        client, headers, event_id = event

        client.post(
            "/me/photo",
            headers=headers,
            files={"photo": ("me.jpg", io.BytesIO(b"alice-profile"), "image/jpeg")},
        )
        client.post(
            f"/events/{event_id}/photos",
            headers=headers,
            files=[
                upload_file("alice.jpg", b"photo-of-alice"),
                upload_file("bob.jpg", b"photo-of-bob"),
                upload_file("both.jpg", b"photo-of-both"),
                upload_file("empty.jpg", b"no-face-here"),
            ],
        )

        data = client.get(f"/events/{event_id}/photos_with_me", headers=headers).json()
        assert data["needs_profile"] is False
        assert data["pending"] == 0
        # Alice appears in alice.jpg and both.jpg, but not bob.jpg/empty.jpg
        assert len(data["photos"]) == 2

    def test_download_all_and_me(self, event):
        client, headers, event_id = event

        client.post(
            "/me/photo",
            headers=headers,
            files={"photo": ("me.jpg", io.BytesIO(b"alice-profile"), "image/jpeg")},
        )
        client.post(
            f"/events/{event_id}/photos",
            headers=headers,
            files=[
                upload_file("alice.jpg", b"photo-of-alice"),
                upload_file("bob.jpg", b"photo-of-bob"),
            ],
        )

        res = client.get(f"/events/{event_id}/download?scope=all", headers=headers)
        assert res.status_code == 200
        assert res.headers["content-type"] == "application/zip"
        with zipfile.ZipFile(io.BytesIO(res.content)) as zf:
            assert len(zf.namelist()) == 2

        res = client.get(f"/events/{event_id}/download?scope=me", headers=headers)
        assert res.status_code == 200
        with zipfile.ZipFile(io.BytesIO(res.content)) as zf:
            assert len(zf.namelist()) == 1

    def test_download_empty_event(self, event):
        client, headers, _ = event
        empty_event = client.post(
            "/events", headers=headers, json={"name": "Empty"}
        ).json()["id"]
        res = client.get(f"/events/{empty_event}/download?scope=all", headers=headers)
        assert res.status_code == 404
