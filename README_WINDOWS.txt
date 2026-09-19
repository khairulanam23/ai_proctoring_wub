================================================================================
AI Proctoring Dataset Collector — Windows User Instructions
================================================================================

Welcome to the AI Proctoring Dataset Collector.
This application allows you to record high-quality, structured still photographs
for research and model development without needing Python or technical tools.

--------------------------------------------------------------------------------
1. BEFORE RUNNING:
--------------------------------------------------------------------------------
1. EXTRACT THE ZIP ARCHIVE:
   If you received this folder inside a .zip file, right-click and choose
   "Extract All..." before running. Do NOT run directly from inside the zip!

2. CAMERA SETUP:
   - Built-in / USB Webcam: Make sure it is plugged in and recognized by Windows.
   - Phone Camera via DroidCam OBS:
     a) Open DroidCam OBS on your PC and connect your phone.
     b) Click "Activate Virtual Camera" in DroidCam OBS so Windows recognizes it.
   - Close any other applications that might be using the camera (Zoom, Teams,
     browser tabs, etc.).

--------------------------------------------------------------------------------
2. HOW TO COLLECT YOUR DATASET:
--------------------------------------------------------------------------------
Step 1: Double-click "DatasetCollector.exe".

Step 2: CAMERA SETUP & FRAMING CHECK:
   - The camera preview will appear.
   - If the wrong camera is selected, click "[C] Switch Camera" or press 'C'
     on your keyboard until your webcam or DroidCam OBS feed appears.
   - Verify framing: Ensure your face, hands, and desk area are clearly visible
     and well-lit.
   - Press ENTER or click "[ENTER] Confirm Camera & Start".

Step 3: CAPTURING ACTIVITIES (45 Activities, 2 Photos Each):
   - For each activity, read the instructions on the top panel:
     * Action  : What physical behavior to perform.
     * Visible : What should be visible in the camera view.
     * Target  : Specific nuance for Photo 1 vs Photo 2.
   - Position yourself according to the instructions.
   - Press SPACEBAR or click "[Capture Photo 1]".
   - The screen will flash white and show "[ PHOTO 1: SAVED ✓ ]".
   - Read the nuance for Photo 2, adjust slightly, and press SPACEBAR again
     or click "[Capture Photo 2]".
   - Once both photos are saved, click "[N] Next >" or press 'N' to proceed.

Step 4: NAVIGATION & RETAKES:
   - Need to retake a photo? Press 'R' or click "[R] Retake" to delete the
     last photo and take it again.
   - Want to go back? Press 'B' or click "[B] < Back".
   - Want to check your output folder? Press 'O' or click "[O] Open Folder".

Step 5: FINISHING:
   - After reviewing the activities, press 'Q' or click "[Q] Finish & Exit".
   - The application will calculate cryptographic SHA-256 checksums and write
     the final manifest.

--------------------------------------------------------------------------------
3. HOW TO SUBMIT YOUR DATASET:
--------------------------------------------------------------------------------
1. When collection finishes, your output folder is located in:
     DatasetOutput\<Participant_ID>\<Session_ID>\
   (For example: DatasetOutput\P001\S001\)

2. Click the "[O] Open Folder" button in the application, or open File Explorer
   and navigate to DatasetOutput\.

3. Right-click the folder (e.g. "S001" or "P001"), choose "Compress to ZIP file".

4. Send or share that .zip file with the dataset administrator.

--------------------------------------------------------------------------------
4. IMPORTANT RULES (WHAT NOT TO DO):
--------------------------------------------------------------------------------
- DO NOT rename, crop, filter, or edit the image files.
- DO NOT edit or modify session_manifest.json.
- DO NOT delete checksum.sha256.
- DO NOT run the application from inside a locked or temporary zip file.

--------------------------------------------------------------------------------
5. TROUBLESHOOTING:
--------------------------------------------------------------------------------
Problem: Camera screen is black or says "CAMERA STREAM DISCONNECTED".
Fix    : Close other camera apps (Zoom, Skype, Chrome). Press 'C' to cycle
         through available camera sources.

Problem: DroidCam OBS is not appearing.
Fix    : Ensure DroidCam OBS is open and the "Virtual Camera" toggle is turned ON
         before launching DatasetCollector.exe. Then press 'C' to select it.

Need technical support?
Send the file in: logs\collector.log to your technical administrator.
================================================================================
