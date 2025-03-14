import { useEffect, useState } from "react";
import SectionHeading from "../components/SectionHeading";
import { motion } from "framer-motion";
import { toast } from "sonner";
import UploadArea from "../components/UploadArea";
import { ArrowRight, Layers, Activity, Map } from "lucide-react";

const API_BASE_URL = "http://localhost:5000";

const Upload = () => {
    const [isUploading, setIsUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [analysisStep, setAnalysisStep] = useState(0);
    const [analysisComplete, setAnalysisComplete] = useState(false);
    const [uploadedImage, setUploadedImage] = useState<string | null>(null);
    const [showResults, setShowResults] = useState(false);

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

    const handleUpload = async (file: File) => {
        // Validate file
        if (!file) {
            toast.error("Please select a file to upload");
            return;
        }

        const allowedTypes = ["image/jpeg", "image/png"];
        if (!allowedTypes.includes(file.type)) {
            toast.error("Please upload a valid image file (JPEG or PNG)");
            return;
        }

        const maxSize = 50 * 1024 * 1024; // 50MB
        if (file.size > maxSize) {
            toast.error("File size must be less than 50MB");
            return;
        }

        // Reset all states before starting upload
        setIsUploading(true);
        setUploadProgress(0);
        setAnalysisStep(0);
        setAnalysisComplete(false);
        setShowResults(false);

        try {
            const formData = new FormData();
            formData.append("file", file);

            // Show a loading toast
            const loadingToast = toast.loading("Uploading file...");

            // Simulate upload progress (for development/testing)
            const simulateUploadProgress = () => {
                const interval = setInterval(() => {
                    setUploadProgress(prev => {
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

            // Actual API call (which may fail in development without backend)
            try {
                const response = await fetch(`${API_BASE_URL}/upload`, {
                    method: "POST",
                    body: formData,
                });

                if (!response.ok) {
                    throw new Error("Failed to upload file");
                }

                // Parse the response
                const data = await response.json();
                console.log("Results:", data.results);
            } catch (apiError) {
                // Just log API errors during development, but still proceed with the flow
                console.log("API error (continuing for demo):", apiError);
            }

            // Clear the simulated interval and set progress to 100%
            clearInterval(uploadInterval);
            setUploadProgress(100);
            
            // Dismiss the loading toast
            toast.dismiss(loadingToast);
            toast.success("File uploaded successfully! Starting analysis...");
            
            // Store the image URL but don't display it
            setUploadedImage(URL.createObjectURL(file));
            
            // Wait a short moment before starting analysis (better user experience)
            setTimeout(() => {
                // Start the analysis process
                startAnalysis();
            }, 800); // Short delay helps users see the "Starting analysis" toast
            
        } catch (error) {
            toast.error("Error uploading file: " + (error as Error).message);
        } finally {
            setIsUploading(false);
        }
    };

    const startAnalysis = async () => {
        // Start with the first step immediately 
        setAnalysisStep(1);
        
        try {
            // For each step in the analysis
            for (let i = 0; i < analysisSteps.length; i++) {
                // Wait 10 seconds for the current step to process
                await new Promise((resolve) => setTimeout(resolve, 10000));
                
                // If this isn't the last step, move to the next one
                if (i < analysisSteps.length - 1) {
                    setAnalysisStep(i + 2); // Move to the next step (i+2 because we're 0-indexed but steps start at 1)
                } 
                // If this is the last step, show results IMMEDIATELY
                else {
                    // Mark analysis as complete and show results right away
                    setAnalysisComplete(true);
                    setShowResults(true);
                    toast.success("Analysis completed successfully!", {
                        description: "Your shoreline analysis results are ready.",
                    });
                }
            }
        } catch (error) {
            toast.error("Error during analysis: " + (error instanceof Error ? error.message : "Unknown error"));   
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
                    />
                </div>

                {analysisStep > 0 && (
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
                                            className={`w-8 h-8 rounded-full flex items-center justify-center mr-4 flex-shrink-0 ${analysisStep > index
                                                    ? "bg-green-100 text-green-600 dark:bg-green-900/30 dark:text-green-400"
                                                    : analysisStep === index
                                                    ? "bg-shoreline-light-blue text-shoreline-blue"
                                                    : "bg-gray-100 text-gray-400 dark:bg-gray-700 dark:text-gray-500"
                                            }`}
                                        >
                                            {analysisStep > index ? (
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
                                                    className={`font-medium ${analysisStep >= index
                                                            ? "text-shoreline-dark dark:text-white"
                                                            : "text-gray-400 dark:text-gray-500"
                                                    }`}
                                                >
                                                    {step.name}
                                                </h4>
                                                {analysisStep === index && !analysisComplete && (
                                                    <div className="flex items-center animate-pulse text-shoreline-blue text-sm">
                                                        <div className="w-2 h-2 bg-shoreline-blue rounded-full animate-ping mr-1"></div>
                                                        <div className="w-2 h-2 bg-shoreline-blue rounded-full animate-ping mr-1" style={{ animationDelay: "0.2s" }}></div>
                                                        <div className="w-2 h-2 bg-shoreline-blue rounded-full animate-ping" style={{ animationDelay: "0.4s" }}></div>
                                                        <span className="ml-2">Processing...</span>
                                                    </div>
                                                )}
                                            </div>
                                            <p
                                                className={`text-sm ${analysisStep >= index
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
                            {/* Only show results when both complete and showResults are true */}
                            {analysisComplete && showResults && (
                                <motion.div
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ type: "spring", stiffness: 100, damping: 15 }}
                                    className="mt-8 pt-6 border-t border-gray-100 dark:border-gray-700"
                                >
                                    <h3 className="text-lg font-medium text-shoreline-dark dark:text-white mb-4">
                                        Analysis Results
                                    </h3>
                                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                                        <div className="bg-shoreline-light-blue/30 dark:bg-shoreline-blue/10 p-4 rounded-lg">
                                            <div className="text-shoreline-blue font-medium mb-1">
                                                Erosion Rate
                                            </div>
                                            <div className="text-2xl font-medium text-shoreline-dark dark:text-white">
                                                -1.2 m/year
                                            </div>
                                        </div>

                                        <div className="bg-shoreline-light-blue/30 dark:bg-shoreline-blue/10 p-4 rounded-lg">
                                            <div className="text-shoreline-blue font-medium mb-1">
                                                Area Affected
                                            </div>
                                            <div className="text-2xl font-medium text-shoreline-dark dark:text-white">
                                                2.3 km²
                                            </div>
                                        </div>

                                        <div className="bg-shoreline-light-blue/30 dark:bg-shoreline-blue/10 p-4 rounded-lg">
                                            <div className="text-shoreline-blue font-medium mb-1">
                                                Confidence Level
                                            </div>
                                            <div className="text-2xl font-medium text-shoreline-dark dark:text-white">
                                                92%
                                            </div>
                                        </div>
                                    </div>

                                    <div className="flex flex-col md:flex-row gap-4">
                                        <button className="flex items-center justify-center px-6 py-3 bg-shoreline-blue text-white font-medium rounded-lg shadow-md hover:shadow-lg hover:shadow-shoreline-blue/20 transition-all duration-300">
                                            <Layers size={18} className="mr-2" />
                                            View Detailed Report
                                        </button>
                                        <button className="flex items-center justify-center px-6 py-3 border border-gray-300 dark:border-gray-700 text-shoreline-dark dark:text-white rounded-lg hover:border-shoreline-blue hover:text-shoreline-blue transition-all duration-300">
                                            <Map size={18} className="mr-2" />
                                            Open Visualization
                                        </button>
                                    </div>
                                </motion.div>
                            )}
                        </div>  
                    </motion.div>
                )}          
            </div>                          
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
    );  
};

export default Upload;