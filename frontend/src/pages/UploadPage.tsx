import { useEffect, useState } from "react";
import SectionHeading from "../components/SectionHeading";
import { motion } from "framer-motion";
import { toast } from "sonner";
import UploadArea from "../components/UploadArea";
import { ArrowRight, Layers, Activity, Map, X } from "lucide-react";
import axios from "axios";
import { useNavigate, useLocation } from 'react-router-dom';

const API_BASE_URL = "http://localhost:5000";

const Upload = () => {
  const navigate = useNavigate();
  const location = useLocation();
  
  // All state declarations at the top
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
  
  // This will run when the component mounts or when the location changes
  useEffect(() => {
    // Scroll to top when component mounts
    window.scrollTo(0, 0);
    
    // Create a cleanup function
    return () => {
      // Clean up states when navigating away
      toast.dismiss();
      setAnalysisComplete(false);
      setShowResults(false);
      setIsUploading(false);
      setIsCancelled(false);
      setError(null);
    };
  }, []);

  const minFiles = 2;
  const maxFiles = 5;

  const [analysisSteps, setAnalysisSteps] = useState([
    {
      name: "Preprocessing image",
      description:
        "Applying radiometric calibration and atmospheric correction",
    },
    {
      name: "Detecting shoreline",
      description:
        "Applying machine learning algorithms to identify water-land boundaries",
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
    // Validate files
    if (!files || files.length < 2 || files.length > 5) {
      toast.error("Please select 2-5 files to upload");
      return;
    }

    // Reset all states before starting upload
    setIsUploading(true);
    setUploadProgress(0);
    setAnalysisStep(0);
    setAnalysisComplete(false);
    setShowResults(false);
    setFiles(files); // Store files for reference
    setError(null);
    setErrorIndex(null);

    try {
      // Create FormData with files - THIS WAS MISSING
      const formData = new FormData();
      
      // Add files with the key 'files' that Flask is expecting
      files.forEach(file => {
        formData.append('files', file);
      });

      // Show a loading toast
      const loadingToast = toast.loading("Uploading files...");

      // Simulate upload progress (for development/testing)
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

        // Parse the upload response that should contain file IDs or paths
        const uploadData = await uploadResponse.json();
        const fileIds = uploadData.fileIds;
        
        // Store the file IDs
        setUploadedFileIds(fileIds);
        
        // Clear the simulated interval and set progress to 100%
        clearInterval(uploadInterval);
        setUploadProgress(100);
        
        // Dismiss the loading toast
        toast.dismiss(loadingToast);
        toast.success("Files uploaded successfully!");
        
        // Store the image URL but don't display it
        setUploadedImage(URL.createObjectURL(files[0]));
        
        // Wait a short moment before showing the first button
        setTimeout(() => {
          setAnalysisStep(0);
          setActiveButton(1); // Show the preprocessing button
        }, 800);
      } catch (apiError) {
        // For demo purposes only - in production, you'd properly handle the error
        const mockFileIds = ['mock-file-1', 'mock-file-2'];
        setUploadedFileIds(mockFileIds);
        
        // Clear the simulated interval and set progress to 100%
        clearInterval(uploadInterval);
        setUploadProgress(100);
        
        // Dismiss the loading toast
        toast.dismiss(loadingToast);
        toast.success("Files uploaded successfully (mock data)");
        
        // Store the image URL but don't display it
        setUploadedImage(URL.createObjectURL(files[0]));
        
        // Start the analysis process with mock file IDs
        setTimeout(() => {
          setAnalysisStep(0);
          setActiveButton(1); // Show the preprocessing button
        }, 800);
      }
    } catch (error) {
      toast.error("Error uploading files: " + (error as Error).message);
    } finally {
      setIsUploading(false);
    }
  };

  const handleCancelAnalysis = () => {
    // For a complete reset, reload the page
    window.location.reload();
    
    // The toast won't be visible after reload, but we'll keep it
    // in case there's a delay before the page refresh
    toast.error("Analysis cancelled");
  };

  // Fix startPreprocessing function
  const startPreprocessing = async (fileIds: string[]) => {
    setError(null);
    setErrorIndex(null);
    setAnalysisStep(1); // Set to step 1
    setActiveButton(null); // Hide button while processing
    
    try {
      // Show a loading toast
      const loadingToast = toast.loading("Preprocessing images...");
      
      const response = await fetch(`${API_BASE_URL}/preprocess`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ fileIds })
      });
      
      const data = await response.json();
      
      // Dismiss loading toast
      toast.dismiss(loadingToast);
      
      if (!response.ok) {
        toast.error("Preprocessing failed");
        
        if (data.invalidImage) {
          setError(data.error || "Problem with image");
          
          // Find index of problematic image
          const errorImgIndex = files.findIndex(
            file => file.name === data.invalidImage
          );
          
          if (errorImgIndex !== -1) {
            setErrorIndex(errorImgIndex);
          }
        } else {
          setError(data.error || 'Failed to preprocess images');
        }
        
        setIsCancelled(true);
        return false;
      }
      
      // Show the next button after successful preprocessing
      toast.success("Preprocessing complete");
      setActiveButton(2); // Show detect shoreline button
      return true;
      
    } catch (err) {
      toast.error("Network error during preprocessing");
      setError('Network error: Could not connect to the preprocessing server');
      setIsCancelled(true);
      return false;
    }
  };

  // Fix detectShoreline function
  const detectShoreline = async (fileIds: string[]) => {
    setError(null);
    setErrorIndex(null);
    setAnalysisStep(2); // Set to step 2
    setActiveButton(null); // Hide button while processing
    
    try {
      // Show a loading toast
      const loadingToast = toast.loading("Detecting shorelines...");
      
      const response = await fetch(`${API_BASE_URL}/detect-shoreline`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ fileIds })
      });
      
      const data = await response.json();
      
      // Dismiss loading toast
      toast.dismiss(loadingToast);
      
      if (!response.ok) {
        toast.error("Shoreline detection failed");
        
        if (data.invalidImage) {
          setError(data.error || "Problem with image");
          
          // Find index of problematic image
          const errorImgIndex = files.findIndex(
            file => file.name === data.invalidImage
          );
          
          if (errorImgIndex !== -1) {
            setErrorIndex(errorImgIndex);
          }
        } else {
          setError(data.error || 'Failed to detect shoreline');
        }
        
        setIsCancelled(true);
        return false;
      }
      
      // Show the next button after successful shoreline detection
      toast.success("Shoreline detection complete");
      setActiveButton(3); // Show measure changes button
      return true;
      
    } catch (err) {
      toast.error("Network error during shoreline detection");
      setError('Network error: Could not connect to the analysis server');
      setIsCancelled(true);
      return false;
    }
  };

  // Fix measureShorelines function
  const measureShorelines = async (fileIds: string[]) => {
    setError(null);
    setErrorIndex(null);
    setAnalysisStep(3); // Set to step 3
    setActiveButton(null); // Hide button while processing
    
    try {
      // Show a loading toast
      const loadingToast = toast.loading("Measuring shoreline changes...");
      
      const response = await fetch(`${API_BASE_URL}/measure-changes`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ fileIds })
      });
      
      // Always parse the response, regardless of status
      const data = await response.json();
      
      // Dismiss loading toast
      toast.dismiss(loadingToast);
      
      // Check if response was not successful
      if (!response.ok) {
        toast.error("Measuring changes failed");
        
        // Set error message from the response
        setError(data.error || 'An error occurred during shoreline analysis');
        
        // If there's an invalid image specified, set the error index
        if (data.invalidImage) {
          const errorIdx = files.findIndex(f => f.name === data.invalidImage);
          if (errorIdx !== -1) {
            setErrorIndex(errorIdx);
          }
        }
        
        setIsCancelled(true);
        return false;
      }
      
      // Success - set results
      toast.success("Change measurements complete");
      
      // Set the results but don't show them yet
      setAnalysisResults(data.models);
      
      // Show the next button after successful measurement
      setActiveButton(4); // Show generate report button
      return true;
      
    } catch (err) {
      toast.error("Network error during measurement");
      setError('Network error: Could not connect to the analysis server');
      setIsCancelled(true);
      return false;
    }
  };

  // Add missing generateReport function
  const generateReport = () => {
    setAnalysisStep(4); // Set to step 4
    setActiveButton(null); // Hide buttons
    
    // Show a loading toast
    const loadingToast = toast.loading("Generating report...");
    
    // Simulate report generation with a delay
    setTimeout(() => {
      toast.dismiss(loadingToast);
      toast.success("Report generated successfully");
      setAnalysisComplete(true);
      setShowResults(true);
    }, 2000);
    
    return true;
  };

  // Function to highlight problematic images
  const highlightProblemImage = (imageName: string) => {
    // Find the problematic image in the files array
    const problemIndex = files.findIndex(file => file.name === imageName);
    
    if (problemIndex >= 0) {
      // You could set some state to highlight this in your UI
      setErrorIndex(problemIndex);
      
      // Scroll to the problematic image
      const imageElement = document.getElementById(`image-${problemIndex}`);
      if (imageElement) {
        imageElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
        imageElement.classList.add('error-highlight');
      }
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
            onClearAll={handleCancelAnalysis} // Add this prop to connect directly
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
                {analysisSteps.map((step, index) => (
                  <div
                    key={index}
                    className="flex items-start"
                  >
                    <div
                      className={`w-8 h-8 rounded-full flex items-center justify-center mr-4 flex-shrink-0 ${
                        (index === 3 && analysisComplete) || analysisStep > index
                          ? "bg-green-100 text-green-600 dark:bg-green-900/30 dark:text-green-400"
                          : analysisStep === index
                          ? "bg-shoreline-light-blue text-shoreline-blue"
                          : "bg-gray-100 text-gray-400 dark:bg-gray-700 dark:text-gray-500"
                      }`}
                    >
                      {(index === 3 && analysisComplete) || analysisStep > index ? (
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
                        <span>{index + 1}</span>
                      )}
                    </div>
                    <div className="flex-1">
                      <div className="flex justify-between">
                        <h4
                          className={`font-medium ${
                            (index === 3 && analysisComplete) || analysisStep >= index
                              ? "text-shoreline-dark dark:text-white"
                              : "text-gray-400 dark:text-gray-500"
                          }`}
                        >
                          {step.name}
                        </h4>
                        {analysisStep === index && 
                          !(index === 3 && analysisComplete) && 
                          !analysisComplete && (
                          <div className="flex items-center animate-pulse text-shoreline-blue text-sm">
                            <div className="w-2 h-2 bg-shoreline-blue rounded-full animate-ping mr-1"></div>
                            <div
                              className="w-2 h-2 bg-shoreline-blue rounded-full animate-ping mr-1"
                              style={{
                                animationDelay:
                                  "0.2s",
                              }}
                            ></div>
                            <div
                              className="w-2 h-2 bg-shoreline-blue rounded-full animate-ping"
                              style={{
                                animationDelay:
                                  "0.4s",
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
                          (index === 3 && analysisComplete) || analysisStep >= index
                            ? "text-shoreline-text dark:text-gray-300"
                            : "text-gray-400 dark:text-gray-500"
                        }`}
                      >
                        {step.description}
                      </p>
                    </div>
                  </div>
                ))}
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
                  
                  {/* Detect Shoreline Button */}
                  {activeButton === 2 && (
                    <motion.button
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      onClick={() => detectShoreline(uploadedFileIds)}
                      className="px-5 py-2.5 bg-shoreline-blue text-white rounded-lg hover:bg-shoreline-blue/90 transition-colors flex items-center"
                    >
                      <span>Detect Shoreline</span>
                      <ArrowRight size={16} className="ml-2" />
                    </motion.button>
                  )}
                  
                  {/* Measure Changes Button */}
                  {activeButton === 3 && (
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
                  {activeButton === 4 && (
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
              {/* Only show results when both complete and showResults are true */}
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
                                  EPR (End
                                  Point Rate)
                                </div>
                                <div className="text-2xl font-medium text-shoreline-dark dark:text-white">
                                  {model.EPR.toFixed(2)}{" "}
                                  m/year
                                </div>
                                <div className="text-xs text-shoreline-text dark:text-gray-400 mt-2">
                                  The rate of
                                  shoreline
                                  change over
                                  time
                                </div>
                              </div>

                              <div className="bg-shoreline-light-blue/30 dark:bg-shoreline-blue/10 p-4 rounded-lg">
                                <div className="text-shoreline-blue font-medium mb-1">
                                  NSM (Net
                                  Shoreline
                                  Movement)
                                </div>
                                <div className="text-2xl font-medium text-shoreline-dark dark:text-white">
                                  {model.NSM}{" "}
                                  m
                                </div>
                                <div className="text-xs text-shoreline-text dark:text-gray-400 mt-2">
                                  Total
                                  movement of
                                  shoreline
                                  position
                                </div>
                              </div>
                            </div>
                          </div>
                        )
                      )}
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
                  <li>Upload images from the same coastal area</li>
                  <li>Make sure images show the same geographic location</li>
                  <li>The images should be from different time periods to measure changes</li>
                  <li>Verify that images have clear shorelines visible</li>
                </ul>
              </div>
              
              <div className="flex justify-end">
                <button 
                  onClick={() => {
                    // Force a complete page reload - this is appropriate here
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
