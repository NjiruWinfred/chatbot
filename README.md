# EduLearn AI Chatbot – Hybrid Adaptive Learning Assistant

## 📖 Overview

EduLearn AI Chatbot is an AI-powered adaptive learning assistant designed to provide personalized lesson recommendations based on student performance data. The system combines a Large Language Model (LLM), local data storage, and caching mechanisms to support both online and offline learning environments.

The chatbot analyzes student learning metrics, including quiz scores, lesson completion rates, time spent on lessons, and question attempt patterns, to recommend whether a student should review content, continue practicing, or advance to the next lesson.

## 🎯 Problem Statement

Many learners struggle because educational systems often provide the same content to all students regardless of individual performance. In environments with limited internet connectivity, accessing AI-powered educational support can be challenging.

EduLearn addresses this challenge by using student performance data to generate personalized recommendations while maintaining functionality through cached data when internet connectivity is unavailable.

## ✨ Features

- AI-powered lesson recommendations using Gemini API
- Personalized learning guidance based on student performance
- Continuous chatbot interaction using Python loops
- Student performance tracking
- MongoDB integration for data storage
- Cached data retrieval for offline functionality
- Adaptive recommendation engine
- Low-connectivity environment support

## 📊 Data Used

The recommendation system utilizes:

- Lesson ID
- Quiz scores per topic
- Number of attempts per question
- Time spent on lessons
- Lesson completion rate

## 🏗️ System Architecture

Student Data
    ↓
MongoDB Database
    ↓
Cache Layer
    ↓
Recommendation Engine
    ↓
Gemini AI Model
    ↓
Personalized Learning Recommendation

### Online Mode

1. Student performance data is retrieved from MongoDB.
2. The recommendation engine processes learning metrics.
3. Gemini AI generates personalized recommendations.
4. Results are returned to the learner.

### Offline Mode

1. Cached student data is retrieved locally.
2. Previously computed insights are used.
3. Recommendations remain available without database access.
4. Learning support continues despite connectivity limitations.

## 🛠️ Technologies Used

- Python
- Gemini API
- MongoDB
- PyMongo
- Local Caching
- Artificial Intelligence
- Data Analysis

## 📂 Project Structure

EduLearn-AI-Chatbot/
│
├── chatbot.py
├── recommendation_engine.py
├── database.py
├── cache.py
├── requirements.txt
├── README.md
└── data/
    └── student_data.json


```bash
git clone <repository-url>
cd EduLearn-AI-Chatbot
