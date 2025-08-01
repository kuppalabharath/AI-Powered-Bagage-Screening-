# AI-Powered-Bagage-Screening-
Implementing Artificial Intelligence-Driven Baggage Screening Measures to Enhance Airport Security and Boost Productivity
A vision-based tool built to assist in airport and public transport security by automatically detecting restricted items in baggage scans using object detection and real-time alerts.


Overview

This project was developed to simulate a smarter, AI-driven approach to baggage screening, aiming to reduce manual errors and improve detection speed at security checkpoints.

The system uses YOLOv11 for real-time object detection on X-ray scanned images, identifies restricted items such as knives, firearms, and tools, and provides instant audio alerts through text-to-speech. Designed with aviation-grade operational needs in mind—speed, reliability, and hands-free operation.


Tech Stack
	•	Python
	•	YOLOv11
	•	OpenCV
	•	Streamlit (for UI)
	•	pyttsx3 (for audio announcements)


 Key Features
	•	 High-Accuracy Detection:
Trained YOLOv11 model on the Hixray X-ray dataset, achieving mAP@0.5 = 74%
	•	 Real-Time Object Detection:
Bounding boxes overlaid on baggage images with label confidence
	•	 Hands-Free Audio Feedback:
Detected items are announced using pyttsx3 to assist in real-time decision-making
	•	 Interactive Interface with Streamlit:
Upload baggage images, run detection, view results with ease
	•	Security-Focused Design:
Built with aviation workflow principles—quick response, low-latency scanning, and ease of use for security personnel
