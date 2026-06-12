"""EventSnap admin CLI.

Usage:
    python cli.py stats                      Show user/event/photo counts
    python cli.py list-users                 List registered users
    python cli.py list-events                List events with photo counts
    python cli.py reset-password EMAIL       Set a new password for a user
    python cli.py encode-pending             Run face encoding for photos that
                                             were never processed
    python cli.py cleanup                    Remove orphaned files and DB rows
"""

import argparse
import getpass
import json
import os
import sys

from sqlalchemy import func, select

from db import Event, Photo, SessionLocal, User, init_db
from security import hash_password

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.environ.get("EVENTSNAP_STORAGE", os.path.join(BASE_DIR, "storage"))
EVENTS_DIR = os.path.join(STORAGE_DIR, "events")


def cmd_stats(db, args):
    users = db.scalar(select(func.count()).select_from(User))
    events = db.scalar(select(func.count()).select_from(Event))
    photos = db.scalar(select(func.count()).select_from(Photo))
    pending = db.scalar(
        select(func.count()).select_from(Photo).where(Photo.encoded == False)  # noqa: E712
    )

    print(f"Users:           {users}")
    print(f"Events:          {events}")
    print(f"Photos:          {photos}")
    print(f"Pending encode:  {pending}")


def cmd_list_users(db, args):
    users = db.scalars(select(User).order_by(User.created_at)).all()
    if not users:
        print("No users registered.")
        return

    for user in users:
        face = "yes" if user.profile_encoding else "no"
        print(f"[{user.id}] {user.name} <{user.email}> face_encoding={face}")


def cmd_list_events(db, args):
    rows = db.execute(
        select(Event, func.count(Photo.id))
        .outerjoin(Photo)
        .group_by(Event.id)
        .order_by(Event.created_at)
    ).all()

    if not rows:
        print("No events.")
        return

    for event, photo_count in rows:
        date = event.event_date or "no date"
        print(f"[{event.id}] {event.name} ({date}) - {photo_count} photo(s)")


def cmd_reset_password(db, args):
    user = db.scalar(select(User).where(User.email == args.email))
    if user is None:
        sys.exit(f"No user with email {args.email}")

    password = getpass.getpass("New password: ")
    if len(password) < 6:
        sys.exit("Password must be at least 6 characters")
    if getpass.getpass("Confirm password: ") != password:
        sys.exit("Passwords do not match")

    user.password_hash = hash_password(password)
    db.commit()
    print(f"Password updated for {user.email}")


def cmd_encode_pending(db, args):
    pending = db.scalars(select(Photo).where(Photo.encoded == False)).all()  # noqa: E712
    if not pending:
        print("Nothing to encode.")
        return

    # Imported lazily: pulls in dlib, which is slow to load
    from app import compute_encodings

    for photo in pending:
        path = os.path.join(EVENTS_DIR, str(photo.event_id), photo.filename)
        if not os.path.exists(path):
            print(f"  photo {photo.id}: file missing, skipping")
            continue

        try:
            encodings = compute_encodings(path)
            photo.encodings = json.dumps([e.tolist() for e in encodings])
            photo.encoded = True
            db.commit()
            print(f"  photo {photo.id}: {len(encodings)} face(s) found")
        except Exception as exc:
            print(f"  photo {photo.id}: failed ({exc})")

    print(f"Processed {len(pending)} photo(s).")


def cmd_cleanup(db, args):
    """Remove photo rows whose file is gone, and files with no DB row."""
    removed_rows = 0
    removed_files = 0

    photos = db.scalars(select(Photo)).all()
    known_files = set()
    for photo in photos:
        path = os.path.join(EVENTS_DIR, str(photo.event_id), photo.filename)
        if os.path.exists(path):
            known_files.add(os.path.abspath(path))
        else:
            db.delete(photo)
            removed_rows += 1
    db.commit()

    if os.path.isdir(EVENTS_DIR):
        for event_id in os.listdir(EVENTS_DIR):
            event_path = os.path.join(EVENTS_DIR, event_id)
            if not os.path.isdir(event_path):
                continue
            for name in os.listdir(event_path):
                path = os.path.abspath(os.path.join(event_path, name))
                if path not in known_files:
                    os.remove(path)
                    removed_files += 1

    print(f"Removed {removed_rows} orphaned DB row(s), {removed_files} orphaned file(s).")


COMMANDS = {
    "stats": cmd_stats,
    "list-users": cmd_list_users,
    "list-events": cmd_list_events,
    "reset-password": cmd_reset_password,
    "encode-pending": cmd_encode_pending,
    "cleanup": cmd_cleanup,
}


def main():
    parser = argparse.ArgumentParser(description="EventSnap admin tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("stats", help="Show user/event/photo counts")
    subparsers.add_parser("list-users", help="List registered users")
    subparsers.add_parser("list-events", help="List events with photo counts")

    reset = subparsers.add_parser("reset-password", help="Set a new password for a user")
    reset.add_argument("email", help="Email of the account to update")

    subparsers.add_parser("encode-pending", help="Encode photos that were never processed")
    subparsers.add_parser("cleanup", help="Remove orphaned files and DB rows")

    args = parser.parse_args()

    init_db()
    db = SessionLocal()
    try:
        COMMANDS[args.command](db, args)
    finally:
        db.close()


if __name__ == "__main__":
    main()
