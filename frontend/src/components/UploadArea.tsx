import { useState, useRef, DragEvent, ChangeEvent } from 'react';
import { Upload, FileType, X, AlertCircle } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface UploadAreaProps {
  onUpload: (file: File) => void;
  isUploading: boolean;
  uploadProgress: number;
}

const UploadArea = ({ onUpload, isUploading, uploadProgress }: UploadAreaProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const allowedFileTypes = ['image/png', 'image/jpeg', 'image/tiff'];
  const maxFileSizeMB = 10;

  const handleDragEnter = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const validateFile = (file: File): boolean => {
    if (!allowedFileTypes.includes(file.type)) {
      setError(`Invalid file type. Please upload PNG, JPEG, or TIFF files.`);
      return false;
    }

    if (file.size > maxFileSizeMB * 1024 * 1024) {
      setError(`File too large. Maximum file size is ${maxFileSizeMB}MB.`);
      return false;
    }

    setError(null);
    return true;
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      if (validateFile(droppedFile)) {
        setFile(droppedFile);
        onUpload(droppedFile);
      }
    }
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      if (validateFile(selectedFile)) {
        setFile(selectedFile);
        onUpload(selectedFile);
      }
    }
  };

  const handleButtonClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleRemoveFile = () => {
    setFile(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      <AnimatePresence mode="wait">
        {!file ? (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
            className={`drop-area ${isDragging ? 'dragging' : ''}`}
            onDragEnter={handleDragEnter}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <div className="flex flex-col items-center">
              <div className="w-16 h-16 rounded-full bg-shoreline-light-blue/50 flex items-center justify-center mb-4">
                <Upload size={28} className="text-shoreline-blue" />
              </div>
              <h3 className="text-lg font-medium text-shoreline-dark dark:text-white mb-2">
                Drag & drop your satellite images
              </h3>
              <p className="text-sm text-shoreline-text dark:text-gray-400 mb-4 text-center">
                Upload Sentinel-2 satellite images of Sri Lankan<br />coastal regions for analysis
              </p>
              
              <div className="flex space-x-4 mb-6">
                <span className="file-type-badge">PNG</span>
                <span className="file-type-badge">JPEG</span>
                <span className="file-type-badge">TIFF</span>
                <span className="file-type-badge">Max: 10MB</span>
              </div>
              
              <button
                onClick={handleButtonClick}
                className="upload-btn"
              >
                Browse Files
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".png,.jpg,.jpeg,.tiff"
                onChange={handleFileChange}
                className="hidden"
              />
            </div>
          </motion.div>
        ) : (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.3 }}
            className="border rounded-lg p-4 bg-white dark:bg-shoreline-dark/30 shadow-sm relative"
          >
            {isUploading && (
              <div className="upload-progress-overlay">
                <div className="w-64 bg-white dark:bg-gray-800 rounded-lg p-4 flex flex-col items-center">
                  <div className="w-full bg-gray-200 dark:bg-gray-700 h-2 rounded-full mb-2 overflow-hidden">
                    <div 
                      className="h-full bg-shoreline-blue transition-all duration-300 ease-out"
                      style={{ width: `${uploadProgress}%` }}
                    />
                  </div>
                  <p className="text-sm font-medium text-shoreline-dark dark:text-white">
                    Uploading... {uploadProgress}%
                  </p>
                </div>
              </div>
            )}
            
            <div className="flex items-center">
              <div className="w-12 h-12 rounded-md bg-shoreline-light-blue/50 flex items-center justify-center mr-4">
                <FileType size={24} className="text-shoreline-blue" />
              </div>
              <div className="flex-1">
                <div className="flex justify-between items-start">
                  <div>
                    <h4 className="text-sm font-medium text-shoreline-dark dark:text-white truncate max-w-xs">
                      {file.name}
                    </h4>
                    <p className="text-xs text-shoreline-text dark:text-gray-400">
                      {(file.size / (1024 * 1024)).toFixed(2)} MB
                    </p>
                  </div>
                  <button
                    onClick={handleRemoveFile}
                    className="p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 transition-colors"
                    disabled={isUploading}
                  >
                    <X size={16} />
                  </button>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      
      {error && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          className="mt-4 p-3 bg-red-50 border border-red-200 rounded-md flex items-center text-sm text-red-800"
        >
          <AlertCircle size={16} className="mr-2 flex-shrink-0" />
          {error}
        </motion.div>
      )}
    </div>
  );
};

export default UploadArea;