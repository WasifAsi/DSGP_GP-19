import { useState, useRef, DragEvent, ChangeEvent, useEffect } from "react";
import { Upload, FileType, X, AlertCircle } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface UploadAreaProps {
  onUpload: (files: File[]) => void;
  isUploading: boolean;
  uploadProgress: number;
  onClear?: () => void; // Add this new prop
  setClearCallback?: (callback: () => void) => void; // Add this new prop
  onClearAll?: () => void; // Add this new prop
}

const UploadArea = ({
  onUpload,
  isUploading,
  uploadProgress,
  onClear,
  setClearCallback,
  onClearAll,
}: UploadAreaProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const [files, setFiles] = useState<File[]>([]); // Changed to array
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const allowedFileTypes = ["image/jpg", "image/jpeg"];
  const maxFileSizeMB = 10;
  const minFiles = 2;
  const maxFiles = 5;

  // Change the regex to look for the date pattern anywhere in the filename
  const fileNameRegex = /\d{4}-\d{2}-\d{2}_[a-zA-Z0-9_-]+/;

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

  // Validate files function with the new file name validation
  const validateFiles = (newFiles: File[]): boolean => {
    // Check number of files
    const totalFiles = files.length + newFiles.length;
    if (totalFiles < minFiles || totalFiles > maxFiles) {
      setError(`Please upload between ${minFiles} and ${maxFiles} images.`);
      return false;
    }

    // Check each file
    for (const file of newFiles) {
      // Validate file type
      if (!allowedFileTypes.includes(file.type)) {
        setError(`Invalid file type. Please upload JPG or JPEG files only.`);
        return false;
      }

      // Get filename without extension for validation
      const fileNameWithoutExt = file.name.split('.').slice(0, -1).join('.');

      // Validate file name format (new more flexible regex)
      const match = fileNameWithoutExt.match(fileNameRegex);
      if (!match) {
        setError(
          `Invalid file name format for "${file.name}". File name must include the pattern YYYY-MM-DD_Location (e.g., "sentinel2_2023-05-20_Colombo.jpg").`
        );
        return false;
      }

      // Validate file size
      if (file.size > maxFileSizeMB * 1024 * 1024) {
        setError(`File "${file.name}" is too large. Maximum file size is ${maxFileSizeMB}MB.`);
        return false;
      }
    }

    setError(null);
    return true;
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const newFiles = Array.from(e.dataTransfer.files);
      if (validateFiles(newFiles)) {
        const updatedFiles = [...files, ...newFiles];
        setFiles(updatedFiles);
        if (updatedFiles.length >= minFiles) {
          onUpload(updatedFiles);
        }
      }
    }
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const newFiles = Array.from(e.target.files);
      if (validateFiles(newFiles)) {
        const updatedFiles = [...files, ...newFiles];
        setFiles(updatedFiles);
        if (updatedFiles.length >= minFiles) {
          onUpload(updatedFiles);
        }
      }
    }
  };

  const handleButtonClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleRemoveFile = (index: number) => {
    const updatedFiles = files.filter((_, i) => i !== index);
    setFiles(updatedFiles);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  // Add clearFiles function near other handlers
  const clearFiles = () => {
    setFiles([]);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  // In the component, add this effect
  useEffect(() => {
    if (onClear) {
      onClear();
    }
  }, [onClear]);

  // Set the callback in the parent when the component mounts
  useEffect(() => {
    if (setClearCallback) {
      setClearCallback(clearFiles);
    }
  }, [setClearCallback]);

  return (
    <div className="w-full max-w-2xl mx-auto">
      <AnimatePresence mode="wait">
        {files.length === 0 ? (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
            className={`drop-area ${isDragging ? "dragging" : ""}`}
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
                Upload 2-5 Sentinel-2 satellite images of Sri Lankan
                <br />
                coastal regions for analysis
                <br />
                <span className="font-medium">File name must include: YYYY-MM-DD_Location</span>
                <br />
                <span className="text-xs">(e.g., "sentinel2_2023-05-20_Colombo.jpg")</span>
              </p>

              <div className="flex space-x-4 mb-6">
                <span className="file-type-badge">JPG</span>
                <span className="file-type-badge">JPEG</span>

                <span className="file-type-badge">Max: 10MB</span>
              </div>

              <button onClick={handleButtonClick} className="upload-btn">
                Browse Files
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".jpg,.jpeg"
                onChange={handleFileChange}
                className="hidden"
                multiple
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
                      style={{
                        width: `${uploadProgress}%`,
                      }}
                    />
                  </div>
                  <p className="text-sm font-medium text-shoreline-dark dark:text-white">
                    Uploading... {uploadProgress}%
                  </p>
                </div>
              </div>
            )}

            <div className="space-y-4">
              {files.map((file, index) => (
                <div key={index} className="flex items-center">
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
                          {((file.size / (1024 * 1024)) as number).toFixed(2)}{" "}
                          MB
                        </p>
                      </div>
                      <button
                        onClick={() => handleRemoveFile(index)}
                        className="p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 transition-colors"
                        disabled={isUploading}
                      >
                        <X size={16} />
                      </button>
                    </div>
                  </div>
                </div>
              ))}

              <div className="flex justify-end space-x-3 mt-4">
                {files.length < maxFiles && !isUploading && files.length < minFiles && (
                  <button
                    onClick={handleButtonClick}
                    className="upload-btn"
                    disabled={isUploading}
                  >
                    Add More Files
                  </button>
                )}

                <button
                  onClick={onClearAll || clearFiles} // Use onClearAll if provided, otherwise just clear files
                  className="px-4 py-2 bg-red-100 text-red-600 hover:bg-red-200 
                          dark:bg-red-900/30 dark:text-red-400 dark:hover:bg-red-900/50 
                          rounded-md transition-colors duration-200 flex items-center space-x-2"
                  disabled={isUploading}
                >
                  <X size={16} />
                  <span>Clear All</span>
                </button>
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
