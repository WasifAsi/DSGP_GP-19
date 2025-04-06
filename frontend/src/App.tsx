"use client";

import { Routes, Route, useLocation } from "react-router-dom";
import Layout from "./components/Layout";
import Index from "./pages/Index";
import UploadPage from "./pages/UploadPage";
import AboutPage from "./pages/AboutPage";
import NotFound from "./pages/NotFound";
import ChatBotPage from "./pages/ChatBotPage";

function App() {
  const location = useLocation();

  return (
    <Routes location={location}>
      <Route path="/" element={<Layout />}>
        <Route index element={<Index />} />
        <Route path="upload" element={<UploadPage />} />
        <Route path="about" element={<AboutPage />} />
        <Route path="chatbot" element={<ChatBotPage />} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  );
}

export default App;
