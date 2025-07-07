# PPEGuard: AI-Powered Personal Protective Equipment Detection System

PPEGuard is an advanced, AI-driven system designed to enhance workplace safety by monitoring Personal Protective Equipment (PPE) compliance in real-time. Leveraging deep learning models trained with NVIDIA RTX 4060 GPU, CUDA, and the RAPIDS library, PPEGuard detects essential PPE items like hardhats, safety vests, gloves, goggles, and masks in live video feeds or uploaded videos. It provides real-time analytics, violation logging, and a user-friendly web interface for both employees and administrators. This project, developed by Vishvvesh Nagappan as part of a Computer Science Engineering initiative at SRM University, focuses on practical AI and machine learning applications in industrial safety.

**Note:** The training code and dataset images are proprietary and are not included in this repository. However, the trained models (CustomCNN, ResNet18, EfficientNet) are provided in the `models/` directory. ResNet18 demonstrated the best performance among the models.

## Table of Contents
* [Project Overview](#project-overview)
* [How It Works](#how-it-works)
    * [1. Deep Learning Models](#1-deep-learning-models)
    * [2. Frontend Functionality & User Experience](#2-frontend-functionality--user-experience)
    * [3. Backend Logic](#3-backend-logic)
* [Tech Stack](#tech-stack)
* [Installation](#installation)
* [Usage](#usage)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)

## Project Overview

PPEGuard addresses the critical issue of PPE non-compliance in high-risk industries, where failure to wear PPE contributes to over 30% of workplace injuries. By integrating state-of-the-art deep learning models with a scalable web application, PPEGuard automates PPE detection, flags violations, and provides actionable insights through an admin dashboard.

## How It Works

PPEGuard operates by processing video frames or images through deep learning models to identify PPE items, ensuring real-time monitoring and compliance.

### 1. Deep Learning Models

PPEGuard employs three distinct deep learning models, all trained on an NVIDIA RTX 4060 GPU utilizing CUDA and the RAPIDS library for accelerated computation. The models were trained on a proprietary dataset of approximately 50,000 images across 12 PPE-related classes (e.g., Hardhat, NO-Hardhat, Safety Vest, Fall-Detected).

**Training Details (Proprietary Code Not Included):**
* **Hardware:** NVIDIA RTX 4060 GPU.
* **Accelerated Libraries:**
    * **CUDA:** For general GPU-accelerated computation.
    * **RAPIDS:** Utilized `cuPy` for NumPy-like GPU operations, `cuDF` for GPU-accelerated dataframes (for preprocessing), and `cuDNN` for optimized deep learning operations.
* **Framework:** PyTorch for model training and inference.
* **Training Techniques:** Mixed Precision Training, Label Smoothing, Balanced Sampling, and Data Augmentation were used to enhance model performance and generalization.

**Models:**
* **CustomCNN:** A custom convolutional neural network optimized for lightweight inference.
* **ResNet18:** A modified ResNet-18 architecture, fine-tuned for PPE classes, offering the best balance of accuracy and speed.
* **EfficientNet:** A scaled EfficientNet model, fine-tuned for high accuracy in complex scenarios.

**Note:** The training code for these models is proprietary and not included in this repository. Only the trained model weights are available in the `models/` directory.

### 2. Frontend Functionality & User Experience

The PPEGuard web interface, built with HTML, Tailwind CSS, and JavaScript, is designed for intuitive navigation and real-time interaction. It comprises several key pages:

* **Login Page:** Authenticates users (employees or admins) with input fields for Employee ID and Department. Includes navigation to the Admin Dashboard. (Image: Screenshot of Login Page)
* **Selection Page:** Allows users to configure detection settings by selecting a target PPE class (e.g., Hardhat, NO-Goggles) and a deep learning model (CustomCNN, ResNet18, or EfficientNet) before starting a session. (Image: Screenshot of Selection Page)
* **Main Detection Page:** The core of the system, displaying real-time PPE detection results.
    * **Live Video Feed:** Displays processed frames from a webcam or uploaded videos with bounding boxes and compliance status.
    * **Real-time PPE Stats:** Visualized as a pie chart using ApexCharts, showing compliance statistics.
    * **Session Controls:** Buttons to "Start Camera," "Stop Camera," "Upload Video" for video file analysis, and "Test Random Samples" to validate model performance on annotated dataset images.
    * **Sample Images Strip:** Displays tested images with bounding boxes for visual verification.
    (Image: Screenshot of Main Detection Page with Live Feed, Pie Chart, and Sample Strip)
* **Admin Dashboard:** Provides administrators with comprehensive analytics and violation logs.
    * **Filters:** Options to filter by department and date range for compliance trends.
    * **Charts:** Compliance trends pie chart and acceptance rate bar chart (ApexCharts).
    * **Employee Table:** Lists Employee ID, Department, Session Count, Acceptance Rate, and Flagged Items, with a "Flagged" button to view detailed violations.
    * **Fault Analysis Section:** Displays a table of violation logs (Violation ID, Employee ID, PPE Item, Date, Timestamp, Duration) with a "Download CSV" option.
    (Image: Screenshot of Admin Dashboard with Charts and Tables)
* **About Us Page:** Introduces the developer, Vishvvesh Nagappan, outlining his background, skills, and contact information. (Image: Screenshot of About Us Page)
* **Description Page:** Details the project's motivation, core functionality, technical backbone (including training process and CUDA/RAPIDS utilization), and future vision. (Image: Screenshot of Description Page)

### 3. Backend Logic

The backend, powered by FastAPI and Python, orchestrates the entire system:

* **User Authentication:** Handles employee and admin logins, storing credentials in a PostgreSQL database.
* **Real-time Communication:** Utilizes Socket.IO for seamless, real-time communication between the frontend and backend, enabling live video feed updates and instant violation alerts.
* **Video Processing:** Captures webcam footage or processes uploaded videos, sending frames to the deep learning models for PPE detection.
* **Inference:** Leverages PyTorch for efficient model inference, with CUDA acceleration on the NVIDIA RTX 4060 GPU.
* **Data Management:** Records non-compliance events with timestamps, employee IDs, and durations into a PostgreSQL database.
* **Analytics Generation:** Processes raw violation data to generate compliance trends, acceptance rates, and detailed violation logs for the Admin Dashboard.
* **API Endpoints:** Provides robust API endpoints for login, model/class selection, video upload, sample testing, and fetching administrative statistics and fault logs.

## Tech Stack

### Training Tech Stack (Proprietary Training Code Not Included)
* **CUDA:** GPU acceleration for training.
* **RAPIDS:**
    * `cuPy`: GPU-accelerated NumPy-like operations.
    * `cuDF`: GPU-accelerated dataframes for preprocessing.
    * `cuDNN`: Optimized deep learning primitives.
* **PyTorch:** Deep learning framework for model training.
* **OpenCV:** Image preprocessing and augmentation.
* **Python:** Core programming language.

### Application Tech Stack
* **Backend:**
    * **FastAPI:** High-performance web framework.
    * **Python-SocketIO:** Real-time communication for video streaming.
    * **PostgreSQL:** Database for employee data, sessions, and violations.
    * **Psycopg2:** PostgreSQL adapter for Python.
    * **OpenCV:** Real-time image processing.
    * **PyTorch:** Model inference.
    * **Pillow:** Image handling.
    * **NumPy:** Numerical computations.
* **Frontend:**
    * **HTML/CSS/JavaScript:** Core web technologies.
    * **Tailwind CSS:** Utility-first CSS framework for responsive design.
    * **Axios:** HTTP client for API requests.
    * **Socket.IO-Client:** Real-time video feed updates.
    * **ApexCharts:** Interactive charts for visualization.
* **Deployment:**
    * **Uvicorn:** ASGI server for FastAPI.
    * **NVIDIA RTX 4060:** GPU for inference, leveraging CUDA.

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/beast27000/PPEGuard.git](https://github.com/beast27000/PPEGuard.git)
    cd PPEGuard
    ```

2.  **Set Up Conda Environment:**
    ```bash
    conda create -n cuda-env python=3.9
    conda activate cuda-env
    ```

3.  **Install Dependencies:**
    ```bash
    pip install fastapi uvicorn python-socketio==5.11.3 opencv-python torch torchvision psycopg2-binary numpy pillow
    # For GPU support (CUDA 11.7 example - adjust if your CUDA version differs)
    pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu117](https://download.pytorch.org/whl/cu117)
    ```

4.  **Set Up PostgreSQL:**
    * Install PostgreSQL and create a database:
        ```sql
        psql -U postgres -c "CREATE DATABASE ppe_detection;"
        ```
    * Update `DB_PARAMS` in `original_code.py` (or your main backend file) with your PostgreSQL credentials.

5.  **Download Static Files:**
    ```bash
    cd static
    curl -o apexcharts.min.js [https://cdn.jsdelivr.net/npm/apexcharts@3.53.0/dist/apexcharts.min.js](https://cdn.jsdelivr.net/npm/apexcharts@3.53.0/dist/apexcharts.min.js)
    curl -o axios.min.js [https://cdn.jsdelivr.net/npm/axios@1.7.7/dist/axios.min.js](https://cdn.jsdelivr.net/npm/axios@1.7.7/dist/axios.min.js)
    curl -o socket.io.min.js [https://cdn.jsdelivr.net/npm/socket.io-client@4.7.5/dist/socket.io.min.js](https://cdn.jsdelivr.net/npm/socket.io-client@4.7.5/dist/socket.io.min.js)
    # Replace this with your actual profile image or remove if not used:
    # magick convert -size 150x150 xc:gray vishvvesh_profile.png 
    echo console.log('Admin password script loaded'); > admin_password.js # This line might need adjustment or removal if not truly used
    cd .. # Go back to root directory
    ```

6.  **Place Trained Models:**
    * Copy the provided `CustomCNN.pth`, `ResNet18.pth`, and `EfficientNet.pth` files to the `models/` directory.

## Usage

1.  **Start the Server:**
    ```bash
    # Navigate to your backend directory, e.g.:
    cd "C:\PPE Detection Cuda\FULL_RAPIDS_USE\CLAUDE\FINAL_MODEL\ppe_detection_web" 
    python server.py
    ```

2.  **Access the Application:**
    * Open `http://localhost:8000/` in your web browser.
    * Log in with an Employee ID and Department.
    * Select a PPE class and model, then proceed to detection.
    * Use the admin dashboard for analytics and violation logs.

3.  **Test Features:**
    * Upload videos or use the webcam for real-time detection.
    * Test random samples to validate model performance.
    * Export violation logs as CSV from the admin dashboard.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for suggestions. Focus areas include:

* Enhancing model accuracy with additional training data.
* Optimizing CUDA-based inference for other GPU architectures.
* Adding new PPE classes or features.

**Note:** The proprietary training code and dataset cannot be modified or shared.

## License

This project is licensed under the MIT License, excluding the proprietary trained models and dataset. The models in `models/` are provided for non-commercial use only.

## Contact

* **Email:** vishvvesh@gmail.com
* **LinkedIn:** [linkedin.com/in/vishvvesh-nagappan-4b1760252](https://www.linkedin.com/in/vishvvesh-nagappan-4b1760252)
* **GitHub:** [github.com/beast27000](https://github.com/beast27000)

Thank you for exploring PPEGuard! Let's make workplaces safer together.
