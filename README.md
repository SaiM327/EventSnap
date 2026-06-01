# EventSnap

EventSnap is a full-stack event photo management application that allows users to upload, organize, and search event photos using facial recognition. Users can create event albums, upload photos, manage their profile, and instantly view all event photos that contain them.

## Features

### Event Album Management
- Browse event photo albums
- View all photos within an event
- Responsive gallery layout
- Clean and intuitive user interface

### Photo Uploads
- Upload multiple photos simultaneously
- Automatic image validation
- Secure file storage with unique filenames
- Real-time gallery updates after upload

### Photo Management
- Delete uploaded photos
- Dynamic gallery refresh
- Protected file handling

### User Profiles
- Upload and update profile pictures
- Save user information locally
- Persistent profile settings
- Profile image management

### Facial Recognition
- Generate face embeddings for uploaded photos
- Create embeddings for profile pictures
- Automatically identify photos containing the user
- Personalized "Photos With Me" gallery

## Tech Stack

### Frontend
- HTML5
- CSS3
- JavaScript

### Backend
- FastAPI
- Python

### Computer Vision
- face_recognition
- NumPy

### Storage
- Local file system storage
- JSON-based face embedding storage

## Architecture

```text
Frontend (HTML/CSS/JS)
          |
          v
      FastAPI API
          |
   +------+------+ 
   |      |      |
   v      v      v
Uploads Profile Face Recognition
   |      |      |
   v      v      v
Storage Embeddings Matching
```

## API Endpoints

### Event Photos

| Method | Endpoint | Description |
|----------|----------|-------------|
| GET | `/event1/photos` | Retrieve all event photos |
| POST | `/event1/upload` | Upload event photos |
| DELETE | `/event1/photos/{filename}` | Delete a photo |
| GET | `/event1/photos_with_me` | Retrieve photos containing the user |

### Profile

| Method | Endpoint | Description |
|----------|----------|-------------|
| POST | `/profile/upload` | Upload profile picture |
| GET | `/profile` | Retrieve current profile picture |

### Health Check

| Method | Endpoint | Description |
|----------|----------|-------------|
| GET | `/health` | Verify server status |

## How It Works

### Upload Event Photos
1. User uploads photos to an event album.
2. Images are stored on the server.
3. Face embeddings are generated for every detected face.
4. Embeddings are saved as JSON files alongside each image.

### Create User Profile
1. User uploads a profile picture.
2. A facial embedding is extracted from the image.
3. The embedding is stored for future matching.

### Find Photos With Me
1. User selects the **Photos With Me** tab.
2. The backend compares the user's profile embedding against all event photo embeddings.
3. Matching photos are returned and displayed instantly.

## Project Structure

```text
EventSnap/
│
├── frontend/
│   ├── index.html
│   ├── event1.html
│   ├── profile.html
│   ├── styles.css
│   └── images/
│
├── backend/
│   ├── main.py
│   ├── storage/
│   │   ├── event1/
│   │   └── profile/
│   └── static/
│
└── requirements.txt
```

## Future Improvements

- User authentication and account management
- Cloud storage integration (AWS S3)
- Support for multiple events and organizers
- FAISS-powered face embedding search for scalability
- Event sharing and collaboration
- Mobile application support
- Real-time photo synchronization
- Automatic face clustering and tagging

## Key Engineering Challenges

- Designing a scalable image upload pipeline
- Managing image storage and retrieval efficiently
- Generating and storing facial embeddings
- Implementing face matching across large photo collections
- Building a responsive and intuitive user experience

## Demo Workflow

1. Upload a profile photo.
2. Upload event photos.
3. Facial embeddings are generated automatically.
4. Open the **Photos With Me** tab.
5. Instantly view all event photos containing your face.

