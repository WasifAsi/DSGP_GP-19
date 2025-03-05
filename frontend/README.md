# Shoreline Analysis

A web application for analyzing shoreline changes in Sri Lanka using satellite imagery and machine learning.

## Overview

This project provides tools to detect and monitor coastal erosion patterns using Sentinel-2 satellite imagery. Built with React, TypeScript, and TailwindCSS, it offers an intuitive interface for researchers and environmental agencies to analyze shoreline changes.

## Features

-   🛰️ **Satellite Image Analysis**: Process Sentinel-2 satellite images to detect shorelines
-   🤖 **Machine Learning Detection**: Advanced algorithms for precise shoreline identification
-   📊 **Change Analysis**: Calculate erosion/accretion rates and temporal changes
-   📱 **Responsive Design**: Full mobile and desktop support
-   🌙 **Dark Mode**: Built-in light/dark theme switching

## Getting Started

### Prerequisites

-   Node.js 18.x or higher
-   npm or yarn

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/shoreline-analysis.git
cd shoreline-analysis
```

2. Install dependencies:

```bash
npm install
```

3. Start the development server:

```
npm run dev
```

-   The application will be available at http://localhost:5173

### Building for Production

```
npm run build
```

### Project Structure

```
src/
├── components/     # Reusable UI components
├── pages/         # Page components
├── assets/        # Static assets
└── styles/        # CSS and styling files
```

## Technology Stack
- **Framework**: React 18
- **Language**: TypeScript
- **Styling**: TailwindCSS
- **Animation**: Framer Motion
- **State Management**: React Hooks
- **Routing**: React Router
- **Build Tool**: Vite
- **Notifications**: Sonner