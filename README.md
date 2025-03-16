# Gotcha Trash - Android Unity App

This repository contains an Android app built with Unity 6, using Barracuda inference and OpenCV for detecting and classifying trash items, particularly optimized for identifying plastics. The app leverages Unity 6 with Barracuda inference and OpenCV for image processing.

## Tech Stack:
- Unity 6
- Barracuda Inference (for Unity)
- OpenCV

## Setup & Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Kilo-zolo/gotcha_trash.git
   ```

2. **Install Unity Hub and Unity 6**
   - Download and install [Unity Hub](https://unity.com/download).
   - Ensure Unity 6 is installed via Unity Hub.

2. **Open Project in Unity**
   - Open Unity Hub.
   - Click on **Open Project** and select **Add project from disk**.
   - Choose the cloned repository directory.

## Running the App

### In Unity Editor:
- Simply click the **Play** button at the top-center of Unity to run within the editor.

### On an Android Device:

**Step 1: Android Device Setup**
- Enable developer mode on your Android device:
  - [Enable Developer Mode](https://www.youtube.com/watch?v=zu9oCE9N8H0)
- Turn on USB Debugging (see above video for step-by-step).

### In Unity:
- Go to **File > Build Settings > Player Settings**.
- Under the **Android** tab:
  - Uncheck **Auto Graphics API**.
  - Ensure **API level 29** (Android 10) or higher is selected for minimum API level.
- Connect your Android device via USB.
- Click on **Build and Run**.

Compiling might take a minute, so please be patient.

- [Watch this video](https://www.youtube.com/watch?v=Nb62CE9N8H0&t=0s) for additional guidance.

## Usage

- Point your camera at objects.
- The app will detect trash items (currently optimized for densely packed plastic items).
- Enjoy exploring the object detection capabilities!

## Notes
- Ensure Unity and Android SDKs are properly set up before building.
- Compilation may take a few moments.


