"use client"

import { Routes, Route } from "react-router-dom"
import { AnimatePresence } from "framer-motion"
import Layout from "./components/Layout"
import Index from "./pages/Index"
import UploadPage from "./pages/UploadPage"
import AboutPage from "./pages/AboutPage"
import NotFound from "./pages/NotFound"
import ModelInsightsPage from "./pages/ModelInsightsPage"

function App() {
  return (
    <AnimatePresence mode="wait">
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Index />} />
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="/model-insights" element={<ModelInsightsPage />} />
          <Route path="*" element={<NotFound />} />
        </Route>
      </Routes>
    </AnimatePresence>
  )
}

export default App
