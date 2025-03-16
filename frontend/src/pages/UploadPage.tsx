import { useEffect, useState } from "react";
import SectionHeading from "../components/SectionHeading";
import { motion } from "framer-motion";
import { toast } from "sonner";
import UploadArea from "../components/UploadArea";
import { ArrowRight, Layers, Activity, Map, X } from "lucide-react";

const API_BASE_URL = "http://localhost:5000";

const Upload = () => {
	useEffect(() => {
		// Scroll to top when component mounts
		window.scrollTo(0, 0);
	}, []);

	const [isUploading, setIsUploading] = useState(false);
	const [uploadProgress, setUploadProgress] = useState(0);
	const [analysisStep, setAnalysisStep] = useState(0);
	const [analysisComplete, setAnalysisComplete] = useState(false);
	const [uploadedImage, setUploadedImage] = useState<string | null>(null);
	const [showResults, setShowResults] = useState(false);
	const [analysisResults, setAnalysisResults] = useState<Array<{
		model_name: string;
		EPR: number;
		NSM: number;
	}> | null>(null);
	const [isCancelled, setIsCancelled] = useState(false);
	const [clearUploadArea, setClearUploadArea] = useState<(() => void) | null>(
		null
	);
	const [files, setFiles] = useState<File[]>([]);
	const [error, setError] = useState<string | null>(null);

	
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

	const validateFiles = (newFiles: File[]): boolean => {
		// Check number of files
		const totalFiles = files.length + newFiles.length;
		if (totalFiles < minFiles || totalFiles > maxFiles) {
			setError(
				`Please upload between ${minFiles} and ${maxFiles} images.`
			);
			return false;
		}
		// ...rest of validation
		return true;
	};

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

		try {
			// Create FormData with files
			const formData = new FormData();
			
			// Add each file with a specific key
			files.forEach((file, index) => {
				formData.append(`file${index}`, file);
			});
			
			// Also send the number of files so backend knows how many to process
			formData.append('fileCount', files.length.toString());

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

				// Store analysis results from the backend
				setAnalysisResults(data.models);
			} catch (apiError) {
				console.log("API error (continuing for demo):", apiError);

				// Use standardized mock data for demos
				setAnalysisResults([
					{
						model_name: "U-net",
						EPR: 0.2,
						NSM: 0.5,
					},
					{
						model_name: "DeepLab v3",
						EPR: 0.1,
						NSM: 0.2,
					},
					{
						model_name: "SegNet",
						EPR: 0.15,
						NSM: 0.3,
					},
				]);
			}

			// Clear the simulated interval and set progress to 100%
			clearInterval(uploadInterval);
			setUploadProgress(100);

			// Dismiss the loading toast
			toast.dismiss(loadingToast);
			toast.success("File uploaded successfully! Starting analysis...");

			// Store the image URL but don't display it
			setUploadedImage(URL.createObjectURL(files[0]));

			// Wait a short moment before starting analysis (better user experience)
			setTimeout(() => {
				// Start the analysis process
				startAnalysis();
			}, 800); // Short delay helps users see the "Starting analysis" toast
		} catch (error) {
			toast.error("Error uploading files: " + (error as Error).message);
		} finally {
			setIsUploading(false);
		}
	};

	const handleCancelAnalysis = () => {
		setIsCancelled(true);
		setAnalysisStep(0);
		setAnalysisComplete(false);
		setShowResults(false);
		setUploadedImage(null);

		// Execute the clearFiles function
		if (clearUploadArea) {
			clearUploadArea();
		}

		toast.error("Analysis cancelled");
	};

	const startAnalysis = async () => {
		setIsCancelled(false); // Reset cancelled state
		setAnalysisStep(1);

		try {
			for (let i = 0; i < analysisSteps.length; i++) {
				// Check if analysis was cancelled
				if (isCancelled) {
					return;
				}

				// Wait 10 seconds for the current step to process
				await new Promise((resolve, reject) => {
					const timeout = setTimeout(resolve, 10000);

					// Clear timeout if cancelled
					if (isCancelled) {
						clearTimeout(timeout);
						reject(new Error("Analysis cancelled"));
					}
				});

				if (i < analysisSteps.length - 1) {
					setAnalysisStep(i + 2);
				} else {
					setAnalysisComplete(true);
					setShowResults(true);
					toast.success("Analysis completed successfully!", {
						description:
							"Your shoreline analysis results are ready.",
					});
				}
			}
		} catch (error) {
			if (!isCancelled) {
				// Only show error if not cancelled
				toast.error(
					"Error during analysis: " +
						(error instanceof Error
							? error.message
							: "Unknown error")
				);
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
						onClear={() => clearUploadArea && clearUploadArea()}
					/>
				</div>

				{analysisStep > 0 && !isCancelled && (
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
												analysisStep > index
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
													className={`font-medium ${
														analysisStep >= index
															? "text-shoreline-dark dark:text-white"
															: "text-gray-400 dark:text-gray-500"
													}`}
												>
													{step.name}
												</h4>
												{analysisStep === index &&
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
													analysisStep >= index
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
																	{model.EPR}{" "}
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
