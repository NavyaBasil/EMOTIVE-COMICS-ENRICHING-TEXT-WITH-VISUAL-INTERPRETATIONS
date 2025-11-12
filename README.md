<b>Emotive Comics: Enriching Text with Visual Interpretations</b>


This repository contains a pioneering, AI-powered web application designed to automatically transform text-based narratives (or voice input) into engaging, multi-panel comic strips. The game provides a novel solution to the time-consuming and skill-intensive process of conventional comic production by using state-of-the-art Stable Diffusion XL (SDXL) models and control mechanisms to ensure character consistency and emotional depth across scenes. The goal is to make digital storytelling accessible to a broad audience, including writers, educators, and hobbyists.

<b>Features</b>


The project's core functionality is **AI-Powered Visual Synthesis**, generating high-quality, comic-style images using the Stable Diffusion XL (SDXL) model while leveraging a Refiner and including **Flexible Input Methods** for both text and voice narration. To maintain narrative quality, **Character Consistency** is achieved across all panels using ControlNet with OpenPose to stabilize the character's appearance and pose. Once visuals are created, the system handles **Comic Panel Rendering**, automatically assembling the images, adding borders, and overlaying narrative text and speech bubbles with the Pillow (PIL) library. Finally, **Performance Optimization** is vital, utilizing features like `xformers` and model CPU offloading to efficiently manage the high VRAM demands of the large diffusion models.

<b>How to Run</b>

Clone the repository to your local machine.
Ensure you have Python 3.8+ and a dedicated NVIDIA GPU with at least 8GB VRAM installed due to the demanding nature of the AI models.
Ensure you have installed the necessary dependencies.
Run the application using the following command:
python app.py
Follow the on-screen instructions by navigating to http://127.0.0.1:5000 in your web browser. The server will take a few minutes to load the AI models on startup.

<b>Dependencies</b>

You may need to install the dependencies using:
pip install flask torch diffusers transformers pillow numpy controlnet-aux speechrecognition xformers
(Note: Installation of torch and xformers may require specific commands based on your operating system and CUDA version.)
