import { useEffect, useState } from "react";
import SectionHeading from "../components/SectionHeading";
import { motion } from "framer-motion";
import { toast } from "sonner";
import UploadArea from "../components/UploadArea";
import { ArrowRight, Layers, Activity, Map, X, Download } from "lucide-react";
import { useNavigate, useLocation } from 'react-router-dom';

const API_BASE_URL = "http://localhost:5000";

const Upload = () => {
  const navigate = useNavigate();
  const location = useLocation();
  
  // State declarations
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [files, setFiles] = useState<File[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [errorIndex, setErrorIndex] = useState<number | null>(null);
  const [analysisStep, setAnalysisStep] = useState(0);
  const [analysisComplete, setAnalysisComplete] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [uploadedFileIds, setUploadedFileIds] = useState<string[]>([]);
  const [activeButton, setActiveButton] = useState<number | null>(null);
  const [isCancelled, setIsCancelled] = useState(false);
  const [analysisResults, setAnalysisResults] = useState<any[]>([]);
  const [clearUploadArea, setClearUploadArea] = useState<(() => void) | null>(null);
  const [processingStep, setProcessingStep] = useState<number | null>(null);
  
  useEffect(() => {
    window.scrollTo(0, 0);
    
    return () => {
      toast.dismiss();
      setAnalysisComplete(false);
      setShowResults(false);
      setIsUploading(false);
      setIsCancelled(false);
      setError(null);
    };
  }, []);

  const [analysisSteps] = useState([
    {
      name: "Validating shoreline images",
      description: "Verifying that uploaded images contain identifiable shorelines",
    },
    {
      name: "Preprocessing images",
      description: "Applying radiometric calibration and atmospheric correction",
    },
    {
      name: "Making mask images",
      description: "Applying machine learning algorithms to create water-land boundary masks",
    },
    {
      name: "Comparing shoreline patterns",
      description: "Verifying that images show the same geographic location",
    },
    {
      name: "Measuring changes",
      description: "Comparing with historical shoreline positions",
    },
    {
      name: "Generating report",
      description: "Compiling analysis results and visualizations",
    },
  ]);

  const handleUpload = async (files: File[]) => {
    if (!files || files.length != 2) {
      toast.error("Please select exactly 2 files to upload");
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);
    setAnalysisStep(0);
    setAnalysisComplete(false);
    setShowResults(false);
    setFiles(files);
    setError(null);
    setErrorIndex(null);

    try {
      const formData = new FormData();
      
      files.forEach(file => {
        formData.append('files', file);
      });

      const loadingToast = toast.loading("Uploading files...");

      const simulateUploadProgress = () => {
        const interval = setInterval(() => {
          setUploadProgress((prev) => {
            if (prev >= 95) {
              clearInterval(interval);
              return 95;
            }
            return prev + 5;
          });
        }, 200);
        return interval;
      };

      const uploadInterval = simulateUploadProgress();

      try {
        const uploadResponse = await fetch(`${API_BASE_URL}/upload`, {
          method: "POST",
          body: formData,
        });
        
        if (!uploadResponse.ok) {
          throw new Error("Failed to upload files");
        }

        const uploadData = await uploadResponse.json();
        const fileIds = uploadData.fileIds;
        
        setUploadedFileIds(fileIds);
        
        clearInterval(uploadInterval);
        setUploadProgress(100);
        
        toast.dismiss(loadingToast);
        toast.success("Files uploaded successfully!");
        
        setUploadedImage(URL.createObjectURL(files[0]));
        
        setTimeout(() => {
          setAnalysisStep(0);
          setActiveButton(0.5);
        }, 800);
      } catch (apiError) {
        const mockFileIds = ['mock-file-1', 'mock-file-2'];
        setUploadedFileIds(mockFileIds);
        
        clearInterval(uploadInterval);
        setUploadProgress(100);
        
        toast.dismiss(loadingToast);
        toast.success("Files uploaded successfully (mock data)");
        
        setUploadedImage(URL.createObjectURL(files[0]));
        
        setTimeout(() => {
          setAnalysisStep(0);
          setActiveButton(0.5);
        }, 800);
      }
    } catch (error) {
      toast.error("Error uploading files: " + (error as Error).message);
    } finally {
      setIsUploading(false);
    }
  };

  const handleCancelAnalysis = () => {
    window.location.reload();
    toast.error("Analysis cancelled");
  };

  const startPreprocessing = async (fileIds: string[]) => {
    setError(null);
    setErrorIndex(null);
    setAnalysisStep(1);
    setActiveButton(null);
    setProcessingStep(1);
    
    try {
      const loadingToast = toast.loading("Preprocessing images...");
      
      const response = await fetch(`${API_BASE_URL}/preprocess`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ fileIds })
      });
      
      toast.dismiss(loadingToast);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error("Server error:", errorText);
        toast.error("Preprocessing failed");
        setError(`Server error: ${response.status} ${response.statusText}`);
        setIsCancelled(true);
        setProcessingStep(null);
        return false;
      }
      
      let data;
      try {
        data = await response.json();
      } catch (jsonError) {
        console.error("JSON parse error:", jsonError);
        toast.error("Error processing server response");
        setError('Error processing server response');
        setIsCancelled(true);
        setProcessingStep(null);
        return false;
      }
      
      if (data.error) {
        toast.error(data.error);
        
        if (data.invalidImage) {
          const errorImgIndex = files.findIndex(
            file => file.name === data.invalidImage
          );
          
          if (errorImgIndex !== -1) {
            setErrorIndex(errorImgIndex);
          }
        }
        
        setError(data.error || 'Failed to preprocess images');
        setIsCancelled(true);
        setProcessingStep(null);
        return false;
      }
      
      toast.success("Preprocessing complete");
      setProcessingStep(null);
      
      setAnalysisStep(2);
      
      setTimeout(() => {
        setActiveButton(2);
      }, 300);
      
      return true;
      
    } catch (err) {
      console.error("Network or parsing error:", err);
      toast.error("Network error during preprocessing");
      setError('Network error: Could not connect to the preprocessing server');
      setIsCancelled(true);
      setProcessingStep(null);
      return false;
    }
  };

  const detectShoreline = async (fileIds: string[]) => {
    setError(null);
    setErrorIndex(null);
    setAnalysisStep(2);
    setActiveButton(null);
    setProcessingStep(2);
    
    try {
      const loadingToast = toast.loading("Creating mask images...");
      
      const response = await fetch(`${API_BASE_URL}/create-masks`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ fileIds })
      });
      
      const data = await response.json();
      
      toast.dismiss(loadingToast);
      
      if (!response.ok) {
        toast.error("Mask creation failed");
        
        if (data.invalidImage) {
          setError(data.error || "Problem with image");
          
          const errorImgIndex = files.findIndex(
            file => file.name === data.invalidImage
          );
          
          if (errorImgIndex !== -1) {
            setErrorIndex(errorImgIndex);
          }
        } else {
          setError(data.error || 'Failed to create mask images');
        }
        
        setIsCancelled(true);
        setProcessingStep(null);
        return false;
      }
      
      toast.success("Mask images created successfully");
      setProcessingStep(null);
      
      setAnalysisStep(3);
      
      setTimeout(() => {
        setActiveButton(3.5);
      }, 300);
      
      return true;
      
    } catch (err) {
      toast.error("Network error during mask creation");
      setError('Network error: Could not connect to the server');
      setIsCancelled(true);
      setProcessingStep(null);
      return false;
    }
  };

  const compareSegmentations = async (fileIds: string[]) => {
    setError(null);
    setErrorIndex(null);
    setAnalysisStep(3);
    setActiveButton(null);
    setProcessingStep(3);
    
    try {
      const loadingToast = toast.loading("Comparing shoreline patterns...");
      
      const response = await fetch(`${API_BASE_URL}/compare-segmentations`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ fileIds })
      });
      
      const data = await response.json();
      
      toast.dismiss(loadingToast);
      
      if (!response.ok) {
        toast.error("Comparison failed");
        
        if (data.invalidImage) {
          setError(data.error || "Problem with image");
          
          const errorImgIndex = files.findIndex(
            file => file.name === data.invalidImage
          );
          
          if (errorImgIndex !== -1) {
            setErrorIndex(errorImgIndex);
          }
        } else {
          setError(data.error || 'Failed to compare images');
        }
        
        setProcessingStep(null);
        setIsCancelled(true);
        return false;
      }
      
      toast.success(data.message || "Images contain similar shoreline patterns");
      setProcessingStep(null);
      
      setAnalysisStep(4);
      
      setTimeout(() => {
        setActiveButton(4);
      }, 300);
      
      return true;
      
    } catch (err) {
      toast.error("Network error during comparison");
      setError('Network error: Could not connect to the server');
      setProcessingStep(null);
      setIsCancelled(true);
      return false;
    }
  };

  const measureShorelines = async (fileIds: string[]) => {
    setError(null);
    setErrorIndex(null);
    setAnalysisStep(4);
    setActiveButton(null);
    setProcessingStep(4);
    
    try {
      const loadingToast = toast.loading("Measuring shoreline changes...");
      
      const response = await fetch(`${API_BASE_URL}/measure-changes`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ fileIds })
      });
      
      const data = await response.json();
      
      toast.dismiss(loadingToast);
      
      if (!response.ok) {
        toast.error("Measuring changes failed");
        
        setError(data.error || 'An error occurred during shoreline analysis');
        
        if (data.invalidImage) {
          const errorIdx = files.findIndex(f => f.name === data.invalidImage);
          if (errorIdx !== -1) {
            setErrorIndex(errorIdx);
          }
        }
        
        setIsCancelled(true);
        setProcessingStep(null);
        return false;
      }
      
      toast.success("Change measurements complete");
      setProcessingStep(null);
      
      setAnalysisResults(data.models);
      
      setAnalysisStep(5);
      
      setTimeout(() => {
        setActiveButton(5);
      }, 300);
      
      return true;
      
    } catch (err) {
      toast.error("Network error during measurement");
      setError('Network error: Could not connect to the analysis server');
      setIsCancelled(true);
      setProcessingStep(null);
      return false;
    }
  };

  const generateReport = () => {
    setAnalysisStep(5);
    setActiveButton(null);
    setProcessingStep(5);
    
    const loadingToast = toast.loading("Generating report...");
    
    setTimeout(() => {
      setProcessingStep(null);
      toast.dismiss(loadingToast);
      toast.success("Report generated successfully");
      
      setTimeout(() => {
        setAnalysisComplete(true);
        setShowResults(true);
      }, 300);
    }, 2000);
    
    return true;
  };

  const validateShoreline = async (fileIds: string[]) => {
    setError(null);
    setErrorIndex(null);
    setAnalysisStep(0.5);
    setActiveButton(null);
    setProcessingStep(0);
    
    try {
      const loadingToast = toast.loading("Validating shorelines...");
      
      const response = await fetch(`${API_BASE_URL}/validate-shoreline`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ fileIds })
      });
      
      toast.dismiss(loadingToast);
      
      let data;
      try {
        data = await response.json();
      } catch (jsonError) {
        console.error("JSON parse error:", jsonError);
        toast.error("Error processing server response");
        setError('Error processing server response');
        setIsCancelled(true);
        setProcessingStep(null);
        return false;
      }
      
      // Handle the case where images aren't shorelines (returns 200 with warning)
      if (data.isNonShoreline) {
        toast.warning(data.warning || "Some images don't contain shorelines");
        
        if (data.invalidImage) {
          const errorImgIndex = files.findIndex(
            file => file.name === data.invalidImage
          );
          
          if (errorImgIndex !== -1) {
            setErrorIndex(errorImgIndex);
          }
        }
        
        setError(data.warning || 'Some uploaded images are not shorelines');
        setIsCancelled(true);
        setProcessingStep(null);
        return false;
      }
      
      // Regular error handling for actual errors (400 responses)
      if (!response.ok) {
        toast.error("Shoreline validation failed");
        setError(data.error || `Server error: ${response.status}`);
        setIsCancelled(true);
        setProcessingStep(null);
        return false;
      }
      
      if (data.error) {
        toast.error(data.error);
        
        if (data.invalidImage) {
          const errorImgIndex = files.findIndex(
            file => file.name === data.invalidImage
          );
          
          if (errorImgIndex !== -1) {
            setErrorIndex(errorImgIndex);
          }
        }
        
        setError(data.error || 'Failed to validate shoreline images');
        setIsCancelled(true);
        setProcessingStep(null);
        return false;
      }
      
      toast.success("Shoreline validation complete");
      setProcessingStep(null);
      
      setAnalysisStep(1);
      
      setTimeout(() => {
        setActiveButton(1);
      }, 300);
      
      return true;
      
    } catch (err) {
      console.error("Network or parsing error:", err);
      toast.error("Network error during shoreline validation");
      setError('Network error: Could not connect to the validation server');
      setIsCancelled(true);
      setProcessingStep(null);
      return false;
    }
  };

  const downloadResults = async (fileId: string) => {
    try {
      const loadingToast = toast.loading("Preparing download...");
      
      // Create a direct download link to the endpoint
      const downloadUrl = `${API_BASE_URL}/download-results/${fileId}`;
      
      // Use fetch to initiate the download
      const response = await fetch(downloadUrl);
      
      if (!response.ok) {
        toast.dismiss(loadingToast);
        toast.error("Failed to download results");
        console.error("Download failed with status:", response.status);
        return;
      }
      
      // Get the blob from the response
      const blob = await response.blob();
      
      // Get filename from Content-Disposition header if available
      let filename = "shoreline-analysis-results.zip";
      const contentDisposition = response.headers.get('Content-Disposition');
      if (contentDisposition) {
        const match = contentDisposition.match(/filename="(.+)"/);
        if (match && match[1]) {
          filename = match[1];
        }
      }
      
      // Create a URL for the blob
      const url = window.URL.createObjectURL(blob);
      
      // Create a temporary anchor element and trigger download
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      
      // Clean up
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      
      toast.dismiss(loadingToast);
      toast.success("Results downloaded successfully");
    } catch (error) {
      toast.error("Error downloading results: " + (error as Error).message);
      console.error("Download error:", error);
    }
  };

  return (
    <div className="pt-32 pb-24">
      <div className="container mx-auto px-4">
        <SectionHeading
          eyebrow="Upload & Analyze"
          title="Shoreline Change Analysis"
          subtitle="Upload satellite images of Sri Lankan coastal regions to analyze shoreline changes and erosion patterns. Our system will automatically detect shorelines and calculate rates of change."
        />

        <div className="mt-16 max-w-4xl mx-auto">
          <UploadArea
            onUpload={handleUpload}
            isUploading={isUploading}
            uploadProgress={uploadProgress}
            setClearCallback={(clearFn) => setClearUploadArea(() => clearFn)}
            onClearAll={handleCancelAnalysis}
          />
        </div>

        {(analysisStep > 0 || activeButton !== null) && !isCancelled && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="mt-16 max-w-3xl mx-auto"
          >
            <div className="bg-white dark:bg-gray-800/30 rounded-xl shadow-lg p-8">
              <h3 className="text-xl font-medium text-shoreline-dark dark:text-white mb-6">
                Analysis Progress
              </h3>
              <div className="space-y-6">
                {analysisSteps.map((step, index) => {
                  const stepNumber = index + 1;
                  
                  const isCurrentStep = 
                    (index === 0 && analysisStep === 0.5) || 
                    analysisStep === index;
                  
                  const isCompleted = 
                    (index === 0 && analysisStep >= 1) ||
                    (index > 0 && analysisStep > index) || 
                    (index === 5 && analysisComplete);
                  
                  const isProcessing = processingStep === index;
                  
                  return (
                    <div
                      key={index}
                      className="flex items-start"
                    >
                      <div
                        className={`w-8 h-8 rounded-full flex items-center justify-center mr-4 flex-shrink-0 ${
                          isCompleted
                            ? "bg-green-100 text-green-600 dark:bg-green-900/30 dark:text-green-400"
                            : isCurrentStep
                            ? "bg-shoreline-light-blue text-shoreline-blue"
                            : "bg-gray-100 text-gray-400 dark:bg-gray-700 dark:text-gray-500"
                        }`}
                      >
                        {isCompleted ? (
                          <svg
                            className="w-5 h-5"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                            xmlns="http://www.w3.org/2000/svg"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth="2"
                              d="M5 13l4 4L19 7"
                            ></path>
                          </svg>
                        ) : (
                          <span>{stepNumber}</span>
                        )}
                      </div>
                      <div className="flex-1">
                        <div className="flex justify-between">
                          <h4
                            className={`font-medium ${
                              isCompleted || isCurrentStep
                                ? "text-shoreline-dark dark:text-white"
                                : "text-gray-400 dark:text-gray-500"
                            }`}
                          >
                            {step.name}
                          </h4>
                          {isProcessing && (
                            <div className="flex items-center animate-pulse text-shoreline-blue text-sm">
                              <div className="w-2 h-2 bg-shoreline-blue rounded-full animate-ping mr-1"></div>
                              <div
                                className="w-2 h-2 bg-shoreline-blue rounded-full animate-ping mr-1"
                                style={{
                                  animationDelay: "0.2s",
                                }}
                              ></div>
                              <div
                                className="w-2 h-2 bg-shoreline-blue rounded-full animate-ping"
                                style={{
                                  animationDelay: "0.4s",
                                }}
                              ></div>
                              <span className="ml-2">
                                Processing...
                              </span>
                            </div>
                          )}
                        </div>
                        <p
                          className={`text-sm ${
                            isCompleted || isCurrentStep
                              ? "text-shoreline-text dark:text-gray-300"
                              : "text-gray-400 dark:text-gray-500"
                          }`}
                        >
                          {step.description}
                        </p>
                      </div>
                    </div>
                  );
                })}
              </div>
              {analysisStep > 0 && !analysisComplete && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="mt-6 flex justify-end"
                >
                  <button
                    onClick={handleCancelAnalysis}
                    className="px-4 py-2 bg-red-100 text-red-600 hover:bg-red-200 dark:bg-red-900/30 dark:text-red-400 dark:hover:bg-red-900/50 rounded-md transition-colors duration-200 flex items-center space-x-2"
                  >
                    <X size={16} />
                    <span>Cancel Analysis</span>
                  </button>
                </motion.div>
              )}
              {/* Step Buttons */}
              {!isCancelled && (
                <div className="mt-8 flex justify-center">
                  {/* Validate Shoreline Button */}
                  {activeButton === 0.5 && (
                    <motion.button
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      onClick={() => validateShoreline(uploadedFileIds)}
                      className="px-5 py-2.5 bg-shoreline-blue text-white rounded-lg hover:bg-shoreline-blue/90 transition-colors flex items-center"
                    >
                      <span>Validate Shoreline Images</span>
                      <ArrowRight size={16} className="ml-2" />
                    </motion.button>
                  )}
                  
                  {/* Preprocessing Button */}
                  {activeButton === 1 && (
                    <motion.button
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      onClick={() => startPreprocessing(uploadedFileIds)}
                      className="px-5 py-2.5 bg-shoreline-blue text-white rounded-lg hover:bg-shoreline-blue/90 transition-colors flex items-center"
                    >
                      <span>Start Preprocessing</span>
                      <ArrowRight size={16} className="ml-2" />
                    </motion.button>
                  )}
                  
                  {/* Create Mask Images Button */}
                  {activeButton === 2 && (
                    <motion.button
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      onClick={() => detectShoreline(uploadedFileIds)}
                      className="px-5 py-2.5 bg-shoreline-blue text-white rounded-lg hover:bg-shoreline-blue/90 transition-colors flex items-center"
                    >
                      <span>Create Mask Images</span>
                      <ArrowRight size={16} className="ml-2" />
                    </motion.button>
                  )}
                  
                  {/* Compare Shorelines Button */}
                  {activeButton === 3.5 && (
                    <motion.button
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      onClick={() => compareSegmentations(uploadedFileIds)}
                      className="px-5 py-2.5 bg-shoreline-blue text-white rounded-lg hover:bg-shoreline-blue/90 transition-colors flex items-center"
                    >
                      <span>Compare Shoreline Patterns</span>
                      <ArrowRight size={16} className="ml-2" />
                    </motion.button>
                  )}
                  
                  {/* Measure Changes Button */}
                  {activeButton === 4 && (
                    <motion.button
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      onClick={() => measureShorelines(uploadedFileIds)}
                      className="px-5 py-2.5 bg-shoreline-blue text-white rounded-lg hover:bg-shoreline-blue/90 transition-colors flex items-center"
                    >
                      <span>Measure Changes</span>
                      <ArrowRight size={16} className="ml-2" />
                    </motion.button>
                  )}
                  
                  {/* Generate Report Button */}
                  {activeButton === 5 && (
                    <motion.button
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      onClick={() => generateReport()}
                      className="px-5 py-2.5 bg-shoreline-blue text-white rounded-lg hover:bg-shoreline-blue/90 transition-colors flex items-center"
                    >
                      <span>Generate Report</span>
                      <ArrowRight size={16} className="ml-2" />
                    </motion.button>
                  )}
                </div>
              )}
              {/* Results section */}
              {analysisComplete &&
                showResults &&
                analysisResults && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{
                      type: "spring",
                      stiffness: 100,
                      damping: 15,
                    }}
                    className="mt-8 pt-6 border-t border-gray-100 dark:border-gray-700"
                  >
                    <h3 className="text-lg font-medium text-shoreline-dark dark:text-white mb-4">
                      Analysis Results
                    </h3>

                    {/* Model Comparison Table
                    <div className="overflow-x-auto">
                        <table className="min-w-full bg-white dark:bg-gray-800 rounded-lg overflow-hidden">
                            <thead className="bg-gray-50 dark:bg-gray-700">
                                <tr>
                                    <th className="py-3 px-4 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                                        Model
                                    </th>
                                    <th className="py-3 px-4 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                                        EPR (End Point Rate)
                                    </th>
                                    <th className="py-3 px-4 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                                        NSM (Net Shoreline Movement)
                                    </th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                                {analysisResults.map((model, index) => (
                                    <tr key={index} className={index % 2 === 0 ? 'bg-white dark:bg-gray-800' : 'bg-gray-50 dark:bg-gray-700/50'}>
                                        <td className="py-4 px-4 text-sm font-medium text-shoreline-dark dark:text-white">
                                            {model.model_name}
                                        </td>
                                        <td className="py-4 px-4 text-sm text-shoreline-dark dark:text-white">
                                            {model.EPR} m/year
                                        </td>
                                        <td className="py-4 px-4 text-sm text-shoreline-dark dark:text-white">
                                            {model.NSM} m
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div> */}

                    {/* Cards View */}
                    <div className="mt-8 space-y-6">
                      {analysisResults.map(
                        (model, index) => (
                          <div
                            key={index}
                            className="bg-white dark:bg-gray-800/30 rounded-lg shadow p-6 border-l-4 border-shoreline-blue"
                          >
                            <h4 className="text-lg font-medium text-shoreline-dark dark:text-white mb-4">
                              {model.model_name}
                            </h4>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                              <div className="bg-shoreline-light-blue/30 dark:bg-shoreline-blue/10 p-4 rounded-lg">
                                <div className="text-shoreline-blue font-medium mb-1">
                                  EPR (End Point Rate)
                                </div>
                                <div className="text-2xl font-medium text-shoreline-dark dark:text-white">
                                  {model.EPR.toFixed(2)} m/year
                                </div>
                                <div className="text-xs text-shoreline-text dark:text-gray-400 mt-2">
                                  The rate of shoreline change over time
                                </div>
                              </div>

                              <div className="bg-shoreline-light-blue/30 dark:bg-shoreline-blue/10 p-4 rounded-lg">
                                <div className="text-shoreline-blue font-medium mb-1">
                                  NSM (Net Shoreline Movement)
                                </div>
                                <div className="text-2xl font-medium text-shoreline-dark dark:text-white">
                                  {model.NSM} m
                                </div>
                                <div className="text-xs text-shoreline-text dark:text-gray-400 mt-2">
                                  Total movement of shoreline position
                                </div>
                              </div>
                            </div>
                          </div>
                        )
                      )}
                    </div>
                    
                    {/* Download Results Button */}
                    <div className="mt-8 flex justify-center">
                      <button
                        onClick={() => downloadResults(uploadedFileIds[0])}
                        className="px-5 py-3 bg-shoreline-blue text-white rounded-lg hover:bg-shoreline-blue/90 transition-colors flex items-center gap-2 shadow-md"
                      >
                        <Download size={18} />
                        <span>Download Analysis Results</span>
                      </button>
                    </div>
                  </motion.div>
                )}
            </div>
          </motion.div>
        )}

        {error && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50"
          >
            <motion.div 
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ type: "spring", damping: 25, stiffness: 300 }}
              className="bg-white dark:bg-gray-800 rounded-xl shadow-xl p-6 max-w-lg w-full mx-4"
            >
              <div className="flex items-center mb-4">
                <div className="w-10 h-10 rounded-full bg-red-100 dark:bg-red-900/40 flex items-center justify-center mr-3">
                  <X size={20} className="text-red-600 dark:text-red-400" />
                </div>
                <h3 className="text-xl font-medium text-red-600 dark:text-red-400">Analysis Error</h3>
              </div>
              
              <p className="text-shoreline-dark dark:text-white mb-6">
                {error}
              </p>
              
              {errorIndex !== null && files[errorIndex] && (
                <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-100 dark:border-red-900/30">
                  <h4 className="text-sm font-medium text-red-800 dark:text-red-300 mb-2">Problem with image:</h4>
                  <div className="flex items-center">
                    <div className="w-16 h-16 bg-gray-100 dark:bg-gray-700 rounded overflow-hidden mr-3 flex-shrink-0">
                      <img 
                        src={URL.createObjectURL(files[errorIndex])} 
                        alt={files[errorIndex].name}
                        className="w-full h-full object-cover"
                      />
                    </div>
                    <p className="text-sm text-shoreline-text dark:text-gray-300 overflow-hidden text-ellipsis">
                      {files[errorIndex].name}
                    </p>
                  </div>
                </div>
              )}
              
              <div className="bg-gray-50 dark:bg-gray-700/50 p-4 rounded-lg mb-6">
                <h4 className="text-sm font-medium text-shoreline-dark dark:text-white mb-2">
                  How to fix this:
                </h4>
                <ul className="text-sm text-shoreline-text dark:text-gray-300 space-y-1 pl-5 list-disc">
                  <li>Upload exactly 2 images from the same coastal area</li>
                  <li>Make sure images show the same geographic location</li>
                  <li>The images should be from different time periods to measure changes</li>
                  <li>Verify that images have clear shorelines visible</li>
                </ul>
              </div>
              
              <div className="flex justify-end">
                <button 
                  onClick={() => {
                    window.location.reload();
                  }}
                  className="px-5 py-2.5 bg-shoreline-blue hover:bg-shoreline-blue/90 text-white rounded-lg transition-colors duration-200 flex items-center"
                >
                  Try Again
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}

        <div className="mt-24">
          <SectionHeading
            eyebrow="How It Works"
            title="Our Analysis Process"
            subtitle="Understanding how we transform satellite imagery into actionable insights for coastal management"
          />
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mt-12">
            <div className="bg-white dark:bg-shoreline-dark/30 p-6 rounded-xl shadow-md relative">
              <div className="w-12 h-12 rounded-full bg-shoreline-light-blue/50 flex items-center justify-center mb-4">
                <Layers
                  size={24}
                  className="text-shoreline-blue"
                />
              </div>
              <h3 className="text-xl font-medium text-shoreline-dark dark:text-white mb-3">
                Image Pre-processing
              </h3>
              <p className="text-shoreline-text dark:text-gray-300 mb-4">
                We apply advanced corrections to enhance
                satellite imagery quality before analysis,
                including atmospheric correction and cloud
                removal.
              </p>
              <ArrowRight className="absolute right-6 bottom-6 text-shoreline-blue hidden lg:block" />
            </div>
            <div className="bg-white dark:bg-shoreline-dark/30 p-6 rounded-xl shadow-md relative">
              <div className="w-12 h-12 rounded-full bg-shoreline-light-blue/50 flex items-center justify-center mb-4">
                <Activity
                  size={24}
                  className="text-shoreline-blue"
                />
              </div>
              <h3 className="text-xl font-medium text-shoreline-dark dark:text-white mb-3">
                ML Shoreline Detection
              </h3>
              <p className="text-shoreline-text dark:text-gray-300 mb-4">
                Our machine learning algorithms precisely
                identify the land-water boundary with high
                accuracy, even in complex coastal environments.
              </p>
              <ArrowRight className="absolute right-6 bottom-6 text-shoreline-blue hidden lg:block" />
            </div>
            <div className="bg-white dark:bg-shoreline-dark/30 p-6 rounded-xl shadow-md relative">
              <div className="w-12 h-12 rounded-full bg-shoreline-light-blue/50 flex items-center justify-center mb-4">
                <Map
                  size={24}
                  className="text-shoreline-blue"
                />
              </div>
              <h3 className="text-xl font-medium text-shoreline-dark dark:text-white mb-3">
                Change Measurement
              </h3>
              <p className="text-shoreline-text dark:text-gray-300 mb-4">
                We calculate shoreline changes by comparing
                historical data with current images, providing
                insights into erosion and accretion patterns.
              </p>
              <ArrowRight className="absolute right-6 bottom-6 text-shoreline-blue hidden lg:block" />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Upload;
