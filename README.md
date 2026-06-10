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


## Future Improvements

- User authentication and account management
- Cloud storage integration (AWS S3)
- Support for multiple events and organizers
- FAISS-powered face embedding search for scalability
- Event sharing and collaboration
- Mobile application support
- Real-time photo synchronization
- Automatic face clustering and tagging

